from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import Any, List, TypedDict

from langgraph.graph import END, START, StateGraph

from day05.app import answer_with_rag


class GraphState(TypedDict, total=False):
    input: str
    path: str
    candidate: str
    output: str
    last_error: str
    retry_count: int
    steps: int
    max_steps: int
    max_retry: int


def build_parser() -> argparse.ArgumentParser:
    """Day08のCLI引数を定義します（入力と上限設定）。"""
    p = argparse.ArgumentParser(prog="day08")
    p.add_argument("--text", required=True)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-retry", type=int, default=1)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.text:
        raise ValueError("--text is required")
    if not args.text.strip():
        raise ValueError("--text must not be empty")
    if not (1 <= args.max_steps <= 50):
        raise ValueError("--max-steps must be between 1 and 50")
    if not (0 <= args.max_retry <= 5):
        raise ValueError("--max-retry must be between 0 and 5")


def run_graph(*, text: str, max_steps: int, max_retry: int) -> str:
    """失敗時復帰（リトライ/フォールバック）付きのフローを実行します。

    実装ガイド：
    - 失敗パターンを1つ以上作り、復帰パスへ入ることを確認する
      - 例：JSONが壊れる→再生成
      - 例：検索ヒットなし→聞き返し
    - `max_steps` / `max_retry` を上限として必ず反映し、無限ループを防ぐ
    - 上限到達時は明示的に失敗（例外）してよい（mainがexit code=1にする）
    """
    if not text.strip():
        raise ValueError("--text must not be empty")
    if not (1 <= max_steps <= 50):
        raise ValueError("--max-steps must be between 1 and 50")
    if not (0 <= max_retry <= 5):
        raise ValueError("--max-retry must be between 0 and 5")

    def count_step(state: GraphState, node_name: str) -> int:
        steps = state.get("steps", 0) + 1
        if steps > state.get("max_steps", max_steps):
            raise RuntimeError(
                f"最大ステップ数に到達しました（max-steps={max_steps}）"
            )
        logging.info("step=%s/%s node=%s", steps, max_steps, node_name)
        return steps

    def choose_path(state: GraphState) -> dict[str, Any]:
        steps = count_step(state, "choose_path")
        input_text = state.get("input", "")
        json_request = re.search(
            r"json|ｊｓｏｎ|不完全なJSON",
            input_text,
            flags=re.IGNORECASE,
        )
        path = "json" if json_request else "search"
        logging.info("selected path=%s", path)
        return {"path": path, "steps": steps}

    def generate_json(state: GraphState) -> dict[str, Any]:
        steps = count_step(state, "generate_json")
        input_text = state.get("input", "").strip()
        retry_count = state.get("retry_count", 0)

        # The README's sample request deliberately produces one malformed
        # response so the bounded retry path can be exercised deterministically.
        if "不完全なJSONを返して" in input_text and retry_count == 0:
            candidate = '{"title":'
        else:
            candidate = json.dumps(
                {
                    "title": input_text,
                    "tasks": [
                        {
                            "id": "task-1",
                            "description": f"入力内容を確認する: {input_text}",
                            "acceptance_criteria": [
                                "入力内容と完了条件が明確になっている"
                            ],
                        }
                    ],
                    "risks": [],
                },
                ensure_ascii=False,
            )
        logging.info("JSON generation attempt=%s", retry_count + 1)
        return {"candidate": candidate, "last_error": "", "steps": steps}

    def validate_json(state: GraphState) -> dict[str, Any]:
        steps = count_step(state, "validate_json")
        try:
            parsed = json.loads(state.get("candidate", ""))
            _validate_generated_json(parsed)
        except (json.JSONDecodeError, ValueError) as exc:
            error = f"生成JSONの検証に失敗しました: {exc}"
            logging.warning("%s", error)
            return {"last_error": error, "steps": steps}
        return {
            "output": json.dumps(parsed, ensure_ascii=False, indent=2),
            "last_error": "",
            "steps": steps,
        }

    def retry_json(state: GraphState) -> dict[str, Any]:
        steps = count_step(state, "retry_json")
        retry_count = state.get("retry_count", 0) + 1
        logging.info(
            "retrying JSON generation retry=%s/%s reason=%s",
            retry_count,
            max_retry,
            state.get("last_error", "unknown error"),
        )
        return {"retry_count": retry_count, "candidate": "", "steps": steps}

    def search(state: GraphState) -> dict[str, Any]:
        steps = count_step(state, "search")
        result = answer_with_rag(state.get("input", ""))
        no_hits = (
            "- (none)" in result
            or "該当する根拠が見つかりませんでした" in result
        )
        if no_hits:
            error = "検索結果がありませんでした"
            logging.info("%s。フォールバックへ進みます", error)
            return {"last_error": error, "steps": steps}
        return {"output": result, "last_error": "", "steps": steps}

    def fallback(state: GraphState) -> dict[str, Any]:
        steps = count_step(state, "fallback")
        logging.info("branch=fallback reason=%s", state.get("last_error", ""))
        return {
            "output": (
                "該当する情報が見つかりませんでした。"
                "別のキーワードや対象を指定して、もう一度質問してください。"
            ),
            "steps": steps,
        }

    def fail(state: GraphState) -> dict[str, Any]:
        count_step(state, "fail")
        raise RuntimeError(
            f"JSONの生成に{state.get('retry_count', 0)}回再試行しましたが"
            f"成功しませんでした（max-retry={max_retry}）。最終エラー: "
            f"{state.get('last_error', '不明なエラー')}"
        )

    def finish(state: GraphState) -> dict[str, Any]:
        steps = count_step(state, "finish")
        output = state.get("output", "").strip()
        if not output:
            raise RuntimeError("処理結果を生成できませんでした")
        return {"output": output, "steps": steps}

    def route_path(state: GraphState) -> str:
        return state.get("path", "search")

    def route_after_validation(state: GraphState) -> str:
        if not state.get("last_error"):
            return "finish"
        if state.get("retry_count", 0) < state.get("max_retry", max_retry):
            return "retry_json"
        return "fail"

    def route_after_search(state: GraphState) -> str:
        return "fallback" if state.get("last_error") else "finish"

    graph = StateGraph(GraphState)
    graph.add_node("choose_path", choose_path)
    graph.add_node("generate_json", generate_json)
    graph.add_node("validate_json", validate_json)
    graph.add_node("retry_json", retry_json)
    graph.add_node("search", search)
    graph.add_node("fallback", fallback)
    graph.add_node("fail", fail)
    graph.add_node("finish", finish)
    graph.add_edge(START, "choose_path")
    graph.add_conditional_edges(
        "choose_path",
        route_path,
        {"json": "generate_json", "search": "search"},
    )
    graph.add_edge("generate_json", "validate_json")
    graph.add_conditional_edges(
        "validate_json",
        route_after_validation,
        {
            "finish": "finish",
            "retry_json": "retry_json",
            "fail": "fail",
        },
    )
    graph.add_edge("retry_json", "generate_json")
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {"fallback": "fallback", "finish": "finish"},
    )
    graph.add_edge("fallback", "finish")
    graph.add_edge("finish", END)
    graph.add_edge("fail", END)

    result = graph.compile().invoke({
        "input": text.strip(),
        "retry_count": 0,
        "steps": 0,
        "max_steps": max_steps,
        "max_retry": max_retry,
    })
    return result["output"]


def _validate_generated_json(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("JSONのトップレベルはオブジェクトである必要があります")
    if not isinstance(value.get("title"), str) or not value["title"].strip():
        raise ValueError("titleは空でない文字列である必要があります")
    if not isinstance(value.get("tasks"), list):
        raise ValueError("tasksは配列である必要があります")
    for index, task in enumerate(value["tasks"]):
        if not isinstance(task, dict):
            raise ValueError(f"tasks[{index}]はオブジェクトである必要があります")
        for key in ("id", "description", "acceptance_criteria"):
            if key not in task:
                raise ValueError(f"tasks[{index}]に{key}がありません")
    if not isinstance(value.get("risks"), list):
        raise ValueError("risksは配列である必要があります")


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    受講者は `run_graph()` を実装します。ここは引数解析/検証/上限の適用/終了コードを担当します。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        _validate_args(args)
    except Exception as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2

    logging.info("max-steps=%s max-retry=%s", args.max_steps, args.max_retry)

    try:
        out = run_graph(text=args.text, max_steps=args.max_steps, max_retry=args.max_retry)
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
