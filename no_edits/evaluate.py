import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


# Ensure repository root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partial_edits.evaluate import get_bigcodebench as _get_bc  # noqa: E402
from partial_edits.evaluate import check_correctness, PASS  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description="No-Edits Test Evaluator (runs test cases)")
    parser.add_argument("--results_path", type=str, required=True, help="Path to JSONL with generated solutions (from generate_solutions)")
    parser.add_argument("--hard", action="store_true", help="Use bigcodebench-hard subset")
    parser.add_argument("--max_workers", type=int, default=2)
    return parser.parse_args()


def evaluate(results_path: str, hard: bool = False, max_workers: int = 2):
    problems = _get_bc(hard=hard)

    tasks = []
    completion_id_map: dict[str, int] = {}

    with open(results_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Handle JSONL containing a single JSON object with "eval" dict
            if isinstance(item, dict) and "eval" in item and "metrics" in item:
                for tid, rec in item.get("eval", {}).items():
                    if tid in problems:
                        solution = rec.get("solution_normalized_for_tests") or rec.get("solution") or ""
                        cid = completion_id_map.get(tid, 0)
                        completion_id_map[tid] = cid + 1
                        tasks.append((cid, problems[tid], solution, f"no_edits/{tid}"))
                break
            else:
                tid = item.get("task_id")
                if tid not in problems:
                    continue
                solution = item.get("solution_normalized_for_tests") or item.get("solution") or ""
                cid = completion_id_map.get(tid, 0)
                completion_id_map[tid] = cid + 1
                tasks.append((cid, problems[tid], solution, f"no_edits/{tid}"))

    print(f"Evaluating {len(tasks)} generated solutions...")
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(check_correctness, t): t for t in tasks}
        for fut in as_completed(future_to_task):
            try:
                all_results.append(fut.result())
            except Exception as e:
                print(f"Evaluation error: {e}")

    passed = sum(1 for r in all_results if r.get("status") == PASS)
    summary = {
        "evaluated": len(all_results),
        "passed": passed,
        "pass@1": (passed / len(all_results)) if all_results else 0.0,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = get_args()
    evaluate(args.results_path, hard=args.hard, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
