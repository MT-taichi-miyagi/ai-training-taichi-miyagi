from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    """Day09のCLI引数を定義します（suite入力と出力先）。"""
    p = argparse.ArgumentParser(prog="day09")
    p.add_argument("--suite", required=True)
    p.add_argument("--out", default=os.path.join("day09", "result.json"))
    p.add_argument("--timeout-sec", type=int, default=30)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.suite:
        raise ValueError("--suite is required")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be a positive integer")


def load_suite(path: str) -> List[Dict[str, Any]]:
    """テスト入力セットを読み込んでケース配列を返します。

    形式は自由ですが、まずはJSONを推奨します。

    例（suite.json）：
    - `[{"id":"case1","input":"..."}, {"id":"case2","input":"..."}]`

    実装ガイド：
    - `path` を開いて読み、ケース配列（list[dict]）にして返す
    - ケースは最低 `id` と `input` を含む想定
    """
    suite_path = Path(path)
    try:
        with suite_path.open("r", encoding="utf-8") as suite_file:
            suite = json.load(suite_file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"suite is not valid JSON ({suite_path}:{exc.lineno}:{exc.colno}): {exc.msg}"
        ) from exc

    if not isinstance(suite, list) or not suite:
        raise ValueError("suite must be a non-empty JSON array")

    seen_ids: set[str] = set()
    for index, case in enumerate(suite):
        if not isinstance(case, dict):
            raise ValueError(f"suite case at index {index} must be a JSON object")
        case_id = case.get("id")
        case_input = case.get("input")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(
                f"suite case at index {index} must have a non-empty string id"
            )
        if case_id in seen_ids:
            raise ValueError(f"duplicate suite case id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(case_input, str):
            raise ValueError(f"suite case {case_id!r} must have a string input")
        if "expected_exit_code" in case and (
            not isinstance(case["expected_exit_code"], int)
            or isinstance(case["expected_exit_code"], bool)
        ):
            raise ValueError(
                f"suite case {case_id!r} expected_exit_code must be an integer"
            )
        if "expected_output_contains" in case and not isinstance(
            case["expected_output_contains"], str
        ):
            raise ValueError(
                f"suite case {case_id!r} expected_output_contains must be a string"
            )
    return suite


def run_case(case: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    """1ケースを実行して結果dictを返します。

    実装ガイド：
    - ここでは「どのDayの成果物を評価するか」を決めて呼び出してください
      例：Day08のフローを呼ぶ、または最終課題を呼ぶ、など
    - 戻り値は最低限このキーを含めると扱いやすいです
      - `id`: ケースID
      - `passed`: bool
      - `reason`: str（失敗理由）
      - `output`: 任意（実際の出力）

    注意：
    - `timeout_sec` を使って、長時間実行にならないようにしてください
    """
    case_id = case.get("id")
    case_input = case.get("input")
    if not isinstance(case_id, str) or not isinstance(case_input, str):
        return {
            "id": str(case_id) if case_id is not None else "(unknown)",
            "passed": False,
            "reason": "case must contain a string id and input",
            "failure_type": "invalid_case",
            "output": "",
        }
    if timeout_sec <= 0:
        return {
            "id": case_id,
            "passed": False,
            "reason": "timeout_sec must be a positive integer",
            "failure_type": "configuration_error",
            "output": "",
        }

    expected_exit_code = case.get("expected_exit_code", 0)
    expected_output = case.get("expected_output_contains")
    if not isinstance(expected_exit_code, int) or isinstance(expected_exit_code, bool):
        return {
            "id": case_id,
            "passed": False,
            "reason": "expected_exit_code must be an integer",
            "failure_type": "invalid_expectation",
            "output": "",
        }
    if expected_output is not None and not isinstance(expected_output, str):
        return {
            "id": case_id,
            "passed": False,
            "reason": "expected_output_contains must be a string",
            "failure_type": "invalid_expectation",
            "output": "",
        }

    repo_root = Path(__file__).resolve().parent.parent
    command = [
        sys.executable,
        "-m",
        "day08.app",
        "--text",
        case_input,
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    started_at = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except TimeoutExpired as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "id": case_id,
            "passed": False,
            "reason": f"timed out after {timeout_sec} seconds",
            "failure_type": "timeout",
            "output": _decode_subprocess_output(exc.stdout),
            "stderr": _decode_subprocess_output(exc.stderr),
            "duration_ms": elapsed_ms,
        }
    except OSError as exc:
        return {
            "id": case_id,
            "passed": False,
            "reason": f"could not start target application: {exc}",
            "failure_type": "launch_error",
            "output": "",
            "stderr": "",
        }

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    passed = completed.returncode == expected_exit_code
    if passed and expected_output is not None:
        passed = expected_output in completed.stdout

    if passed:
        reason = "completed with expected result"
        failure_type = None
    elif completed.returncode != expected_exit_code:
        reason = (
            f"expected exit code {expected_exit_code}, "
            f"got {completed.returncode}"
        )
        failure_type = "exit_code_mismatch"
    else:
        reason = f"output did not contain expected text: {expected_output!r}"
        failure_type = "output_mismatch"

    return {
        "id": case_id,
        "passed": passed,
        "reason": reason,
        "failure_type": failure_type,
        "output": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "exit_code": completed.returncode,
        "duration_ms": elapsed_ms,
    }


def _decode_subprocess_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace").strip()
    return output.strip()


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    suite読み込み→各ケース実行→結果JSON出力までを制御します。
    受講者は `load_suite()` と `run_case()` を実装します。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        _validate_args(args)
    except Exception as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2

    try:
        suite = load_suite(args.suite)
    except (OSError, ValueError) as e:
        logging.error("%s", e)
        print(str(e), file=sys.stderr)
        return 2

    try:
        results: List[Dict[str, Any]] = []
        ok = 0
        ng = 0

        for case in suite:
            start = time.perf_counter()
            r = run_case(case, timeout_sec=args.timeout_sec)
            r.setdefault("duration_ms", int((time.perf_counter() - start) * 1000))
            passed = bool(r.get("passed", False))
            ok += 1 if passed else 0
            ng += 0 if passed else 1
            results.append(r)

        out_obj = {
            "summary": {"passed": ok, "failed": ng, "total": ok + ng},
            "failure_types": {
                failure_type: sum(
                    1 for result in results
                    if result.get("failure_type") == failure_type
                )
                for failure_type in sorted({
                    result["failure_type"]
                    for result in results
                    if result.get("failure_type")
                })
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print(f"passed={ok} failed={ng} total={ok + ng}")
        return 0
    except Exception as e:
        logging.error("%s", e)
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
