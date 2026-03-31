"""
Recompute similarity metrics on existing eval results JSON files.
Skips test execution - only updates similarity_metrics fields.
"""
import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from utils.extract_utils import extract_function_body

from evaluate import (
    SimilarityCalculator,
    _process_similarity_metrics,
    _calculate_corpus_metrics,
    _compute_difference_with_canonical,
)


def recompute_file(result_path: str) -> bool:
    """Recompute similarity metrics for a single eval results JSON. Returns True on success."""
    with open(result_path, "r") as f:
        results = json.load(f)

    if "eval" not in results or not results["eval"]:
        return False

    bodies_cache = {}

    def get_body_cached(solution: str) -> str:
        if solution not in bodies_cache:
            bodies_cache[solution] = extract_function_body(solution)
        return bodies_cache[solution]

    similarity_calc = SimilarityCalculator(get_body_cached=get_body_cached)

    for task_id, task_data in results["eval"].items():
        similarity_metrics = _process_similarity_metrics(
            {
                "canonical_solution": task_data.get("canonical_solution"),
                "corrupted_solution": task_data.get("corrupted_solution"),
                "solution": task_data.get("solution", ""),
            },
            eval_similarity=True,
            similarity_calc=similarity_calc,
        )
        similarity_metrics["difference_with_canonical"] = _compute_difference_with_canonical(
            similarity_metrics
        )
        task_data["similarity_metrics"] = similarity_metrics

    results["metrics"]["similarity_metrics"] = _calculate_corpus_metrics(results, eval_similarity=True)
    results["metrics"]["eval_similarity"] = True

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Recompute similarity metrics on existing eval results (no test execution)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path", help="Path to a single eval results JSON file")
    group.add_argument("--dir", help="Directory to recursively process all *_eval_results.json")

    args = parser.parse_args()

    if args.path:
        paths = [args.path]
        if not os.path.isfile(args.path):
            print(f"Error: File not found: {args.path}")
            return 1
    else:
        base = Path(args.dir)
        if not base.is_dir():
            print(f"Error: Directory not found: {args.dir}")
            return 1
        paths = sorted(base.rglob("*_eval_results.json"))

    if not paths:
        print("No eval results files found.")
        return 0

    success = 0
    errors = []
    for path in tqdm(paths, desc="Recomputing similarity"):
        path_str = str(path)
        try:
            if recompute_file(path_str):
                success += 1
        except Exception as e:
            print(f"Error processing {path_str}: {e}")
            errors.append(path_str)

    print(f"Done. Processed {success}/{len(paths)} files successfully.")
    print(f"Errors: {errors}")
    return 0 if success == len(paths) else 1


if __name__ == "__main__":
    exit(main())
