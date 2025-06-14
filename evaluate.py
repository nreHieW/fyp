# References: https://github.com/bigcode-project/bigcodebench/tree/main
import json
import os
import sys
import multiprocessing
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from partial_edits.utils.code_utils import untrusted_check
from partial_edits.utils.extract_utils import get_levenshtein_distance, count_diff_lines
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"


def calculate_bleu_score(text1: str, text2: str) -> float:
    """Calculate BLEU score between two code snippets"""
    # Tokenize the code by splitting on whitespace and common delimiters
    tokens1 = text1.replace("\n", " ").replace("\t", " ").split()
    tokens2 = text2.replace("\n", " ").replace("\t", " ").split()

    # Use smoothing function to handle cases where n-gram matches are zero
    smoothing = SmoothingFunction().method1

    # Calculate BLEU score (treating text1 as reference, text2 as hypothesis)
    try:
        bleu_score = sentence_bleu([tokens1], tokens2, smoothing_function=smoothing)
        return bleu_score
    except:
        return 0.0


def calculate_similarity_metrics(text1: str, text2: str) -> Dict[str, float]:
    """Calculate similarity metrics between two code snippets"""
    return {"bleu_score": calculate_bleu_score(text1, text2), "levenshtein_distance": get_levenshtein_distance(text1, text2), "edit_distance": count_diff_lines(text1, text2)}


def load_solutions(sample_path: str):
    """Load solutions from JSONL file or directory

    Expected JSONL format:
    Each line should be a JSON object with the following structure:

    Required fields:
    - task_id (str): The identifier for the BigCodeBench task (e.g., "BigCodeBench/123")
    - solution (str): The complete standalone code solution

    Example format:
    {"task_id": "BigCodeBench/123", "solution": "def solve_problem():\\n    # complete implementation\\n    return result"}

    Notes:
    - One JSON object per line (JSONL format)
    - Samples with task_ids not in BigCodeBench dataset will be skipped during evaluation

    Args:
        sample_path (str): Path to JSONL file containing solutions

    Returns:
        list: List of solution dictionaries with _identifier field added
    """
    solutions = []

    if os.path.isfile(sample_path):
        with open(sample_path, "r") as fp:
            for i, line in enumerate(fp):
                if any(not x.isspace() for x in line):
                    sample = json.loads(line)
                    assert "solution" in sample, "No 'solution' field found!"
                    sample["_identifier"] = f"{sample['task_id']} (line {i+1} in {sample_path})"
                    solutions.append(sample)

    return solutions


def get_bigcodebench():
    dataset_name = "bigcode/bigcodebench-hard"
    dataset = load_dataset(dataset_name, split="v0.1.4")
    return {item["task_id"]: item for item in dataset}


def check_correctness(task_info: Tuple[int, Dict[str, Any], str, str]) -> Dict:
    """Check if a solution is correct

    Args:
        task_info: Tuple of (completion_id, problem, solution, identifier)
    """
    completion_id, problem, solution, identifier = task_info
    status, details = untrusted_check(solution, problem["test"])

    return {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
        "status": status,
        "details": details,
    }


def evaluate(sample_path: str, save_results: bool = True, max_workers: int | None = None, eval_similarity: bool = False):
    assert sample_path is not None, "No samples provided"
    assert sample_path.endswith(".jsonl"), "Samples must be a JSONL file"

    # Set up result paths
    result_path = sample_path.replace(".jsonl", "_eval_results.json")

    # Load problems
    print("Loading BigCodeBench dataset...")
    problems = get_bigcodebench()

    # Initialize results
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval": {},
        "similarity": {} if eval_similarity else None,
    }

    completion_id = Counter()

    print("Preparing solutions for evaluation...")
    solutions = load_solutions(sample_path)

    # Prepare tasks for concurrent execution
    tasks = []
    similarity_data = {}  # Store data for similarity evaluation

    for sample in solutions:
        task_id = sample["task_id"]

        if task_id not in problems:
            continue

        solution = sample["solution"]

        # Store data for similarity evaluation
        if eval_similarity:
            similarity_data[task_id] = {
                "corrupted_solution": sample["corrupted_solution"],
                # take from the samples because the one in the hf dataset is just function body without function signature
                "canonical_solution": sample["canonical_solution"],
                "llm_solution": sample["solution"],
                "sample": sample,
            }

        # Create task tuple for concurrent execution
        task_info = (completion_id[task_id], problems[task_id], solution, sample["_identifier"])
        tasks.append(task_info)
        completion_id[task_id] += 1

    if not tasks:
        print("No valid tasks found to evaluate.")
        return results, {"pass@1": 0.0, "total_problems": len(problems), "evaluated_problems": 0, "passed_problems": 0}

    # Determine number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(tasks))

    print(f"Evaluating {len(tasks)} solutions using {max_workers} workers...")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(check_correctness, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Evaluating") as pbar:
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    all_results.append(result)

                    # # Print failed/timeout results for debugging
                    # if result["status"] == FAIL or result["status"] == TIMEOUT:
                    #     print(f"\n{result['_identifier']}: {result['status']}")
                    #     print(result["details"])
                    #     print()

                except Exception as e:
                    task_info = future_to_task[future]
                    print(f"Error evaluating task {task_info[1]['task_id']}: {e}")

                pbar.update(1)

    # Process results and calculate pass@1 directly
    passed_tasks = 0
    total_tasks = len(all_results)

    for result in all_results:
        task_id = result["task_id"]

        # Store result for saving
        results["eval"][task_id] = {
            "task_id": task_id,
            "solution": result["solution"],
            "canonical_solution": next((s["canonical_solution"] for s in solutions if s["task_id"] == task_id), None),
            "corrupted_solution": next((s["corrupted_solution"] for s in solutions if s["task_id"] == task_id), None),
            "status": result["status"],
            "details": result["details"],
        }

        if result["status"] == PASS:
            passed_tasks += 1

    # Calculate similarity metrics if requested
    if eval_similarity:
        print("Calculating similarity metrics...")
        results["similarity"] = {}

        for task_id, sim_data in tqdm(similarity_data.items(), desc="Computing similarity"):
            corrupted_solution = sim_data["corrupted_solution"]
            canonical_solution = sim_data["canonical_solution"]
            llm_solution = sim_data["llm_solution"]

            similarity_results = {}

            # 1. Corrupted vs Canonical
            if corrupted_solution and canonical_solution:
                similarity_results["corrupted_vs_canonical"] = calculate_similarity_metrics(canonical_solution, corrupted_solution)

            # 2. LLM vs Corrupted
            if llm_solution and corrupted_solution:
                similarity_results["llm_vs_corrupted"] = calculate_similarity_metrics(corrupted_solution, llm_solution)

            # 3. LLM vs Canonical
            if llm_solution and canonical_solution:
                similarity_results["llm_vs_canonical"] = calculate_similarity_metrics(canonical_solution, llm_solution)

            results["similarity"][task_id] = similarity_results

    # Calculate average similarity metrics
    avg_similarity = {}
    if eval_similarity and results["similarity"]:
        # Initialize accumulators for each metric type
        metric_sums = {}
        metric_counts = {}

        for task_id, task_similarities in results["similarity"].items():
            for comparison_type, metrics in task_similarities.items():
                if comparison_type not in metric_sums:
                    metric_sums[comparison_type] = {}
                    metric_counts[comparison_type] = {}

                for metric_name, value in metrics.items():
                    if metric_name not in metric_sums[comparison_type]:
                        metric_sums[comparison_type][metric_name] = 0
                        metric_counts[comparison_type][metric_name] = 0

                    metric_sums[comparison_type][metric_name] += value
                    metric_counts[comparison_type][metric_name] += 1

        # Calculate averages
        for comparison_type in metric_sums:
            avg_similarity[comparison_type] = {}
            for metric_name in metric_sums[comparison_type]:
                avg_similarity[comparison_type][metric_name] = metric_sums[comparison_type][metric_name] / metric_counts[comparison_type][metric_name]

    # Calculate pass@1
    pass_at_1 = passed_tasks / total_tasks if total_tasks > 0 else 0.0

    # Metadata
    metrics = {
        "pass@1": pass_at_1,
        "total_problems": len(problems),
        "evaluated_problems": total_tasks,
        "passed_problems": passed_tasks,
        "eval_similarity": eval_similarity,
        "avg_similarity": avg_similarity if eval_similarity else None,
    }

    # Print results
    print(f"\nBigCodeBench-Hard")
    print(f"pass@1: {pass_at_1:.3f}")
    print(f"Evaluated {total_tasks}/{len(problems)} problems")
    print(f"Passed {passed_tasks}/{total_tasks} problems")

    if eval_similarity:
        print(f"Similarity metrics computed for {len(results['similarity'])} problems")

        if avg_similarity:
            print("\nAverage Similarity Metrics:")
            for comparison_type, metrics_dict in avg_similarity.items():
                print(f"  {comparison_type}:")
                for metric_name, avg_value in metrics_dict.items():
                    print(f"    {metric_name}: {avg_value:.4f}")

    # Save results
    if save_results:
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {result_path}")

    return results, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simplified BigCodeBench Evaluator")
    parser.add_argument("--sample_path", help="Path to JSONL file containing solutions")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of worker processes (default: number of CPU cores)")
    parser.add_argument("--eval_similarity", action="store_true", default=False, help="Evaluate similarity metrics (BLEU, Levenshtein, edit distance)")

    args = parser.parse_args()

    try:
        results, metrics = evaluate(sample_path=args.sample_path, save_results=not args.no_save, max_workers=args.max_workers, eval_similarity=args.eval_similarity)
        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
