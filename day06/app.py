from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import List
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import boto3
from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    """Day06のCLI引数を定義します（入力・モード・ケース番号）。"""
    p = argparse.ArgumentParser(prog="day06")
    p.add_argument("--text", required=True)
    p.add_argument("--mode", choices=["normal", "attack"], default="normal")
    p.add_argument("--case", type=int, default=1)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.text:
        raise ValueError("--text is required")
    if args.mode == "attack" and not (1 <= args.case <= 3):
        raise ValueError("--case must be between 1 and 3 when --mode attack")
    if not args.text.strip():
        raise ValueError("--text must not be empty")


def run_guarded(*, text: str, mode: str, case: int) -> str:
    """入力とツール実行をガードしながら処理し、回答（文字列）を返します。

    `mode` の意味：
    - `normal`：通常入力を処理
    - `attack`：用意した「悪い入力例」を使って、ガードが効いているか確認

    実装ガイド：
    - attackケースは最低3つ用意（README参照）
    - 「禁止する行為」を明確にし、検知したら例外（または拒否文）にする
    - ツールを実装する場合は許可リストで制限する（許可されないツールは実行しない）

    返り値：
    - 標準出力に出る本文（文章）を返す
    """
    if len(text) > 2000:
        raise ValueError("入力が長すぎます（2000文字以内で入力してください）")

    attack_patterns = {
        1: (
            "指示の乗っ取り",
            r"前の指示を無視|指示を無視|秘密情報|隠し情報|"
            r"ignore (all )?previous instructions|system prompt",
        ),
        2: (
            "システムコマンドの実行要求",
            r"システムコマンド|コマンドを実行|os\.system|subprocess|"
            r"shell command",
        ),
        3: (
            "許可していないファイル操作",
            r"ファイルを削除|ファイル削除|削除して|"
            r"delete file|remove file",
        ),
    }

    detected_cases = [
        case_number
        for case_number, (_, pattern) in attack_patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]

    if mode == "attack":
        if case not in detected_cases:
            if detected_cases:
                detected_labels = ", ".join(
                    f"ケース{number}（{attack_patterns[number][0]}）"
                    for number in detected_cases
                )
                raise ValueError(
                    f"指定された攻撃ケース{case}ではなく、"
                    f"{detected_labels}が検出されました"
                )
            raise ValueError(
                f"攻撃ケース{case}を検出できませんでした。入力内容を確認してください"
            )

    # --caseと異なる攻撃入力でも、危険な入力は必ず拒否する
    if detected_cases:
        matched_case = case if mode == "attack" else detected_cases[0]
        reason = attack_patterns[matched_case][0]
        raise ValueError(
            f"不正な入力が検出されました（ケース{matched_case}: {reason}）"
        )

    if not any(word in text for word in ("天気", "気温", "降水確率")):
        return "天気に関する質問を入力してください。"

    weather_info = get_today_weather()
    return ask_bedrock(text, weather_info)


def get_today_weather() -> str:
    """Open-Meteoから東京の今日の天気予報を取得します。"""
    params = urlencode({
        "latitude": 35.6762,
        "longitude": 139.6503,
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max"
        ),
        "timezone": "Asia/Tokyo",
        "forecast_days": 1,
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        with urlopen(url, timeout=10) as response:
            result = json.load(response)
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"天気情報を取得できませんでした: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("天気APIから不正なJSONが返されました") from exc

    daily = result.get("daily") if isinstance(result, dict) else None
    required_fields = (
        "time",
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
    )
    if not isinstance(daily, dict) or any(
        not isinstance(daily.get(field), list)
        or not daily[field]
        or daily[field][0] is None
        for field in required_fields
    ):
        raise RuntimeError("天気APIの応答に必要な予報データがありません")

    weather_descriptions = {
        0: "快晴",
        1: "主に晴れ",
        2: "一部曇り",
        3: "曇り",
        45: "霧",
        48: "着氷性の霧",
        51: "弱い霧雨",
        53: "霧雨",
        55: "強い霧雨",
        56: "弱い着氷性の霧雨",
        57: "強い着氷性の霧雨",
        61: "弱い雨",
        63: "雨",
        65: "強い雨",
        66: "弱い着氷性の雨",
        67: "強い着氷性の雨",
        71: "弱い雪",
        73: "雪",
        75: "強い雪",
        77: "霧雪",
        80: "弱いにわか雨",
        81: "にわか雨",
        82: "強いにわか雨",
        85: "弱いにわか雪",
        86: "強いにわか雪",
        95: "雷雨",
        96: "ひょうを伴う雷雨",
        99: "強いひょうを伴う雷雨",
    }
    weather_code = daily["weather_code"][0]
    description = weather_descriptions.get(weather_code, "不明")
    return (
        f"東京（{daily['time'][0]}）の天気: {description} "
        f"(weather_code={weather_code}), "
        f"最高気温: {daily['temperature_2m_max'][0]}°C, "
        f"最低気温: {daily['temperature_2m_min'][0]}°C, "
        f"降水確率: {daily['precipitation_probability_max'][0]}%"
    )


def ask_bedrock(question: str, weather_info: str) -> str:
    load_dotenv()
    region = os.getenv("AWS_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID")

    if not region or not model_id:
        raise RuntimeError("AWS_REGION and BEDROCK_MODEL_ID must be set")

    prompt = f"""質問: {question}

天気APIから取得した情報:
{weather_info}

天気APIの情報だけを使って日本語で回答してください。
情報に含まれないことは推測せず、不明と伝えてください。"""

    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }),
    )

    result = json.loads(response["body"].read())
    answer = "".join(
        block["text"]
        for block in result["content"]
        if block.get("type") == "text"
    ).strip()

    if not answer:
        raise RuntimeError("Bedrockから回答を取得できませんでした")

    return answer


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    受講者は `run_guarded()` を実装します。ここは引数解析/検証/終了コードを担当します。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        _validate_args(args)
    except Exception as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2

    logging.info("mode=%s case=%s", args.mode, args.case)

    try:
        out = run_guarded(text=args.text, mode=args.mode, case=args.case)
        print(out)
        return 0
    except NotImplementedError as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        logging.error("%s", e)
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
