from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List
import os
import boto3
from dotenv import load_dotenv

def build_parser() -> argparse.ArgumentParser:
    """Day03のCLI引数を定義します（要件文とリトライ回数）。"""
    p = argparse.ArgumentParser(prog="day03")
    p.add_argument("--requirements", required=True)
    p.add_argument("--max-retry", type=int, default=1)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.requirements:
        raise ValueError("--requirements is required")
    if not (0 <= args.max_retry <= 3):
        raise ValueError("--max-retry must be between 0 and 3")


def generate_json(requirements: str) -> str:
    """要件文字列から、JSON文字列（本文のみ）を生成して返します。

    この関数を実装すると、`python -m day03.app --requirements ...` が動くようになります。

    実装ガイド：
    - LLMに「JSONだけを返す」ように強く指示する
    - `title` / `tasks` / `risks` を必ず含める
    - `tasks` は配列で、各要素に `id` / `description` / `acceptance_criteria` を含める
    - 返す文字列は JSON として `json.loads()` できる必要がある

    注意：
    - 余計な前置き/後置きの文章を混ぜない
    - 壊れやすいので、プロンプトは短く・形式を固定する

    # TODO(TRAINEE): Generate a JSON string that passes validate_json().
    raise NotImplementedError("Implement JSON generation")
    """
    region = os.getenv("AWS_REGION")    
    model_id = os.getenv("BEDROCK_MODEL_ID")
    if not region or not model_id:
        raise RuntimeError("AWS_REGION and BEDROCK_MODEL_ID must be set in environment variables")

    prompt = f"""次の要件をタスクに分解し、有効なJSONオブジェクトだけを返し、```jsonなどのコードフェンスや説明文を付けないでください。
Markdownのコードフェンスや説明文は付けないでください。
必ず title、tasks、risks を含めてください。
tasks は配列とし、各要素に id、description、
acceptance_criteria を含めてください。

要件: {requirements}"""

    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": [{ "type": "text", "text": prompt }],
                }
            ],
        }),
    )
    try:
        result = json.loads(response['body'].read())
    except json.JSONDecodeError as e:
        raise RuntimeError("Bedrock API response body was not valid JSON") from e
    text = "".join(
        block["text"]
        for block in result["content"]
        if block.get("type") == "text"
    ).strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    if not text:
        raise RuntimeError("Bedrock response did not text")
    return text

def validate_json(text: str) -> Dict[str, Any]:
    """生成結果のJSONを検証します（必須キーと型）。"""
    try :
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:300]
        raise ValueError(
            f"LLM output was not valid JSON"
            f"(line {e.lineno}, column {e.colno}: {e.msg}); "
            f"response starts with: {preview}"
        ) from e
    for key in ("title", "tasks", "risks"):
        if key not in obj:
            raise ValueError(f"missing key: {key}")
    if not isinstance(obj.get("tasks"), list):
        raise ValueError("tasks must be a list")
    return obj


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    JSON生成→検証→（失敗時は再生成）までを制御します。受講者は `generate_json()` を実装します。
    """
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except Exception as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2

    last_err: Exception | None = None
    for _ in range(args.max_retry + 1):
        try:
            text = generate_json(args.requirements)
            validate_json(text)
            print(text)
            return 0
        except NotImplementedError as e:
            logging.error(str(e))
            print(str(e), file=sys.stderr)
            return 1
        except Exception as e:
            last_err = e

    msg = str(last_err) if last_err else "validation failed"
    logging.error(msg)
    print(msg, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
