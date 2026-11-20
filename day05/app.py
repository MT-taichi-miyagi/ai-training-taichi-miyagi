from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from typing import List
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Day05のCLI引数を定義します（質問文）。"""
    p = argparse.ArgumentParser(prog="day05")
    p.add_argument("--question", required=True)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.question:
        raise ValueError("--question is required")


def answer_with_rag(question: str) -> str:
    """RAGで質問に回答し、指定フォーマットのテキストを返します。

    この関数を実装すると、`python -m day05.app --question ...` が動くようになります。

    要件（READMEの出力フォーマット）：
    - 標準出力に次の形で出すための文字列を返す
      1) `Answer:` 行
      2) `Sources:` 行
      3) `- <URL or ファイル名> (excerpt: "...")` を最低1件（ヒットなしなら `- (none)`）

    実装ガイド：
    - `day05/data/` 配下の `.txt` を読み込み、検索対象とする
    - 最初は単純なキーワード検索でもOK（高品質でなくてよい）
    - ヒットがない場合の挙動を必ず実装する
    """
    data_dir = Path(__file__).parent / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"データフォルダーがありません: {data_dir}")

    files = sorted(data_dir.glob("*.txt"))

    # Punctuation and whitespace should not count as search terms.
    normalized_question = question.strip().lower()
    if not normalized_question:
        raise ValueError("質問を入力してください")

    query_parts = [
        re.sub(r"[^\w]", "", part)
        for part in normalized_question.split()
    ]
    keywords = list(dict.fromkeys(
        part[index:index + 2]
        for part in query_parts
        for index in range(len(part) - 1)
    ))
    if not keywords:
        keywords = [part for part in query_parts if part]

    # A single shared bigram is too weak to count as a relevant document.
    min_score = max(1, math.ceil(len(keywords) * 0.5))
    if len(keywords) > 1:
        min_score = max(2, min_score)

    matches = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # 質問の検索語が多く含まれる行を、根拠として選ぶ
        best_line = max(
            lines,
            key=lambda line: sum(
                keyword in line.lower() for keyword in keywords
            ),
            default="",
        )
        score = sum(keyword in best_line.lower() for keyword in keywords)

        if score >= min_score:
            excerpt = best_line[:200].replace('"', '\\"')
            source = file_path.relative_to(data_dir.parent.parent).as_posix()
            matches.append((score, source, excerpt))

    if not matches:
        return (
            "Answer: 該当する根拠が見つかりませんでした。\n"
            "Sources:\n"
            "- (none)"
        )

    matches.sort(key=lambda match: match[0], reverse=True)
    _, _, best_excerpt = matches[0]

    source_lines = [
        f'- {source} (excerpt: "{excerpt}")'
        for _, source, excerpt in matches[:3]
    ]

    return (
        f"Answer: 資料には「{best_excerpt}」とあります。\n"
        "Sources:\n"
        + "\n".join(source_lines)
    )


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    受講者は `answer_with_rag()` を実装します。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except Exception as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2

    try:
        out = answer_with_rag(args.question)
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
