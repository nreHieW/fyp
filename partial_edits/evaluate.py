# References: https://github.com/bigcode-project/bigcodebench/tree/main
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Tuple, List
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.extract_utils import extract_function_body, standardize_code_formatting
from utils.similarity_utils import (
    get_levenshtein_distance,
    count_diff_lines,
    get_structured_diff,
    get_rouge_scores,
    get_meteor_score,
    get_chrf_score,
    get_codebleu_score,
    calculate_bleu_score,
    calculate_corpus_bleu_score,
    calculate_corpus_codebleu_score,
)
import requests

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

EVALUATOR_URL = "http://localhost:8000"


def calculate_diffs(corrupted_solution: str, canonical_solution: str, llm_solution: str) -> Dict[str, List[Dict[str, Any]]]:
    diffs = {}
    # 1. Corrupted vs Canonical
    if corrupted_solution and canonical_solution:
        diffs["corrupted_vs_canonical"] = get_structured_diff(corrupted_solution, canonical_solution)

    # 2. Corrupted vs LLM
    if corrupted_solution and llm_solution:
        diffs["corrupted_vs_llm"] = get_structured_diff(corrupted_solution, llm_solution)

    # 3. Canonical vs LLM
    if canonical_solution and llm_solution:
        diffs["canonical_vs_llm"] = get_structured_diff(canonical_solution, llm_solution)

    return diffs


def _compute_raw_metrics(text1: str, text2: str, ignore_comments: bool = False) -> Dict[str, float]:
    """Compute raw similarity metrics for two text strings"""
    metrics = {
        "bleu_score": calculate_bleu_score(text1, text2, ignore_comments=ignore_comments),
        "levenshtein_distance": get_levenshtein_distance(text1, text2, ignore_comments=ignore_comments),
        "edit_distance": count_diff_lines(text1, text2, ignore_comments=ignore_comments),
        "meteor_score": get_meteor_score(text1, text2, ignore_comments=ignore_comments),
        "chrf_score": get_chrf_score(text1, text2, ignore_comments=ignore_comments),
    }

    rouge_scores = get_rouge_scores(text1, text2, ignore_comments=ignore_comments)
    metrics.update(rouge_scores)

    codebleu_scores = get_codebleu_score(text1, text2, ignore_comments=ignore_comments)
    metrics.update(codebleu_scores)

    return metrics


def _normalize_metrics(raw_metrics: Dict[str, float], text1: str, text2: str, prefix: str = "", ignore_comments: bool = False) -> Dict[str, float]:
    """Normalize distance metrics and add prefix if provided"""
    if ignore_comments:
        text1 = standardize_code_formatting(text1)
        text2 = standardize_code_formatting(text2)

    max_len = max(len(text1), len(text2))
    max_lines = max(len(text1.splitlines()), len(text2.splitlines()))

    result = {}

    # Add all raw metrics with prefix
    for key, value in raw_metrics.items():
        result[f"{prefix}{key}"] = value

    # Add normalized versions of distance metrics
    result[f"{prefix}normalized_levenshtein"] = raw_metrics["levenshtein_distance"] / max_len if max_len > 0 else 0.0
    result[f"{prefix}normalized_edit_distance"] = raw_metrics["edit_distance"] / max_lines if max_lines > 0 else 0.0

    return result


def calculate_similarity_metrics_with_comments_only(original: str, edited: str) -> Dict[str, float]:
    """Calculate similarity metrics with comments only (no comment stripping)"""
    raw_metrics = _compute_raw_metrics(original, edited, ignore_comments=False)
    return _normalize_metrics(raw_metrics, original, edited, ignore_comments=False)


def calculate_body_similarity_metrics_with_comments_only(original: str, edited: str, get_body_cached=None) -> Dict[str, float]:
    """Calculate similarity metrics for function bodies with comments only"""
    # Use cached extraction if available, otherwise extract directly
    if get_body_cached:
        original_body = get_body_cached(original)
        edited_body = get_body_cached(edited)
    else:
        original_body = extract_function_body(original).strip()
        edited_body = extract_function_body(edited).strip()

    # If body extraction fails, return zero metrics
    if not original_body or not edited_body:
        # Create zero metrics for simplified structure
        zero_metrics = {
            "bleu_score": 0.0,
            "levenshtein_distance": 0.0,
            "edit_distance": 0.0,
            "meteor_score": 0.0,
            "chrf_score": 0.0,
            "rouge1_fmeasure": 0.0,
            "rouge2_fmeasure": 0.0,
            "rougeL_fmeasure": 0.0,
            "codebleu": 0.0,
        }
        return _normalize_metrics(zero_metrics, "", "", "body_", ignore_comments=False)

    # With comments only
    raw_metrics = _compute_raw_metrics(original_body, edited_body, ignore_comments=False)
    return _normalize_metrics(raw_metrics, original_body, edited_body, "body_", ignore_comments=False)


def calculate_similarity_metrics(original: str, edited: str) -> Dict[str, float]:
    """Calculate similarity metrics both with and without comments"""
    # With comments
    raw_metrics_with_comments = _compute_raw_metrics(original, edited, ignore_comments=False)
    metrics_with_comments = _normalize_metrics(raw_metrics_with_comments, original, edited, ignore_comments=False)

    # Without comments
    raw_metrics_no_comments = _compute_raw_metrics(original, edited, ignore_comments=True)
    metrics_no_comments = _normalize_metrics(raw_metrics_no_comments, original, edited, ignore_comments=True)

    # Combine both sets with appropriate prefixes
    combined_metrics = {}

    # Add metrics with comments (no prefix for backward compatibility)
    combined_metrics.update(metrics_with_comments)

    # Add metrics without comments with "no_comments_" prefix
    for key, value in metrics_no_comments.items():
        combined_metrics[f"no_comments_{key}"] = value

    return combined_metrics


def calculate_body_similarity_metrics(original: str, edited: str, get_body_cached=None) -> Dict[str, float]:
    """Calculate similarity metrics for function bodies with optional caching, both with and without comments"""

    # Use cached extraction if available, otherwise extract directly
    if get_body_cached:
        original_body = get_body_cached(original)
        edited_body = get_body_cached(edited)
    else:
        original_body = extract_function_body(original).strip()
        edited_body = extract_function_body(edited).strip()

    # If body extraction fails, return zero metrics
    if not original_body or not edited_body:
        # Create zero metrics for simplified structure
        zero_metrics = {
            "bleu_score": 0.0,
            "levenshtein_distance": 0.0,
            "edit_distance": 0.0,
            "meteor_score": 0.0,
            "chrf_score": 0.0,
            "rouge1_fmeasure": 0.0,
            "rouge2_fmeasure": 0.0,
            "rougeL_fmeasure": 0.0,
            "codebleu": 0.0,
        }
        with_comments = _normalize_metrics(zero_metrics, "", "", "body_", ignore_comments=False)
        no_comments = _normalize_metrics(zero_metrics, "", "", "body_no_comments_", ignore_comments=True)
        return {**with_comments, **no_comments}

    # With comments
    raw_metrics_with_comments = _compute_raw_metrics(original_body, edited_body, ignore_comments=False)
    metrics_with_comments = _normalize_metrics(raw_metrics_with_comments, original_body, edited_body, "body_", ignore_comments=False)

    # Without comments
    raw_metrics_no_comments = _compute_raw_metrics(original_body, edited_body, ignore_comments=True)
    metrics_no_comments = _normalize_metrics(raw_metrics_no_comments, original_body, edited_body, "body_no_comments_", ignore_comments=True)

    return {**metrics_with_comments, **metrics_no_comments}


def load_solutions(sample_path: str):
    """Load solutions from JSONL file or directory

    Expected JSONL format:
    Each line should be a JSON object with the following structure:

    Required fields:
    - task_id (str): The identifier for the BigCodeBench task (e.g., "BigCodeBench/123")
    - solution (str): The complete standalone code solution

    Optional fields:
    - prompt (str): The prompt that was used to generate the solution
    - llm_response (str): The raw LLM response before extraction
    - canonical_solution (str): The canonical solution for the task
    - corrupted_solution (str): The corrupted solution for the task

    Example format:
    {"task_id": "BigCodeBench/123", "solution": "def solve_problem():\\n    # complete implementation\\n    return result", "prompt": "Solve this problem...", "llm_response": "Here is the solution:\\n\\n```python\\ndef solve_problem():\\n    # complete implementation\\n    return result\\n```"}

    Notes:
    - One JSON object per line (JSONL format)
    - Samples with task_ids not in BigCodeBench dataset will be skipped during evaluation
    - The prompt and llm_response fields will be saved in the evaluation results for viewing in the UI

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


def get_bigcodebench(hard=False):
    if hard:
        dataset_name = "bigcode/bigcodebench-hard"
    else:
        dataset_name = "bigcode/bigcodebench"
    dataset = load_dataset(dataset_name, split="v0.1.4", cache_dir="data")
    return {item["task_id"]: item for item in dataset}


def check_correctness(task_info: Tuple[int, Dict[str, Any], str, str]) -> Dict:
    completion_id, problem, solution, identifier = task_info

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{EVALUATOR_URL}/evaluate", json={"completion_id": completion_id, "problem": problem, "solution": solution, "identifier": identifier})
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:  # Not the last attempt
                wait_time = 10
                print(f"Evaluation failed for {identifier}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e  # Re-raise on final attempt


def evaluate(sample_path: str, save_results: bool = True, max_workers: int = 8, eval_similarity: bool = False, hard: bool = False):
    assert sample_path is not None, "No samples provided"
    assert sample_path.endswith(".jsonl"), "Samples must be a JSONL file"

    sample_dir = os.path.dirname(sample_path)
    sample_filename = os.path.basename(sample_path)
    results_dir = os.path.join(sample_dir, "results")
    result_filename = sample_filename.replace(".jsonl", "_eval_results.json")
    result_path = os.path.join(results_dir, result_filename)

    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    dataset_name = "bigcode/bigcodebench-hard" if hard else "bigcode/bigcodebench"
    print(f"Loading BigCodeBench dataset: {dataset_name}...")
    problems = get_bigcodebench(hard=hard)

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval": {},
        "metrics": {},
    }

    completion_id = Counter()

    print("Preparing solutions for evaluation...")
    solutions = load_solutions(sample_path)

    tasks = []

    for sample in solutions:
        task_id = sample["task_id"]

        if task_id not in problems:
            continue

        solution = sample["solution"]

        task_info = (completion_id[task_id], problems[task_id], solution, sample["_identifier"])
        tasks.append(task_info)
        completion_id[task_id] += 1

    if not tasks:
        print("No valid tasks found to evaluate.")
        return results, {"pass@1": 0.0, "total_problems": len(problems), "evaluated_problems": 0, "passed_problems": 0}

    print(f"Evaluating {len(tasks)} solutions using {max_workers} workers...")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(check_correctness, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Evaluating") as pbar:
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    task_info = future_to_task[future]
                    print(f"Error evaluating task {task_info[1]['task_id']}: {e}")

                pbar.update(1)

    passed_tasks = 0
    total_tasks = len(all_results)

    print("Processing results and calculating diffs...")

    for result in all_results:
        task_id = result["task_id"]

        original_sample = next((s for s in solutions if s["task_id"] == task_id), None)
        canonical_solution = original_sample["canonical_solution"] if original_sample else None
        corrupted_solution = original_sample["corrupted_solution"] if original_sample else None
        llm_solution = result["solution"]

        diffs = calculate_diffs(corrupted_solution or "", canonical_solution or "", llm_solution or "")

        individual_similarity_metrics = {}
        if eval_similarity:
            bodies_cache = {}

            def get_body_cached(solution: str) -> str:
                if solution not in bodies_cache:
                    bodies_cache[solution] = extract_function_body(solution).strip()
                return bodies_cache[solution]

            # Helper function to compute all metrics for a pair (with and without comments)
            def compute_pair_metrics_full(original: str, edited: str) -> Dict[str, float]:
                full_metrics = calculate_similarity_metrics(original, edited)
                body_metrics = calculate_body_similarity_metrics(original, edited, get_body_cached)
                return {**full_metrics, **body_metrics}

            # Helper function to compute only with-comments metrics for a pair
            def compute_pair_metrics_comments_only(original: str, edited: str) -> Dict[str, float]:
                full_metrics = calculate_similarity_metrics_with_comments_only(original, edited)
                body_metrics = calculate_body_similarity_metrics_with_comments_only(original, edited, get_body_cached)
                return {**full_metrics, **body_metrics}

            # Compute similarity pairs - only canonical vs corrupted gets comments-only metrics
            if corrupted_solution and canonical_solution:
                individual_similarity_metrics["corrupted_vs_canonical"] = compute_pair_metrics_comments_only(canonical_solution, corrupted_solution)

            # LLM comparisons get full metrics (with and without comments)
            if llm_solution and corrupted_solution:
                individual_similarity_metrics["llm_vs_corrupted"] = compute_pair_metrics_full(corrupted_solution, llm_solution)

            if llm_solution and canonical_solution:
                individual_similarity_metrics["llm_vs_canonical"] = compute_pair_metrics_full(canonical_solution, llm_solution)

        results["eval"][task_id] = {
            "task_id": task_id,
            "solution": result["solution"],
            "canonical_solution": canonical_solution,
            "corrupted_solution": corrupted_solution,
            "status": result["status"],
            "details": result["details"],
            "diffs": diffs,
            "similarity_metrics": individual_similarity_metrics if eval_similarity else None,
            "prompt": original_sample.get("prompt", "") if original_sample else "",
            "llm_response": original_sample.get("llm_response", "") if original_sample else "",
            "token_usage": original_sample.get("token_usage", {}) if original_sample else {},
        }

        if result["status"] == PASS:
            passed_tasks += 1

    # Calculate average similarity metrics from individual sample metrics
    avg_similarity = {}
    if eval_similarity:
        metric_sums = {}
        metric_counts = {}

        # Use the individual similarity metrics we calculated for each sample
        for task_id, task_data in results["eval"].items():
            if task_data.get("similarity_metrics"):
                for comparison_type, metrics in task_data["similarity_metrics"].items():
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

        # Calculate corpus-level BLEU and CodeBLEU metrics and add them to avg_similarity
        for comparison_type in ["corrupted_vs_canonical", "llm_vs_corrupted", "llm_vs_canonical"]:
            references = []
            predictions = []

            for task_id, task_data in results["eval"].items():
                if task_data.get("similarity_metrics") and comparison_type in task_data["similarity_metrics"]:
                    if comparison_type == "corrupted_vs_canonical":
                        ref = task_data.get("canonical_solution", "")
                        pred = task_data.get("corrupted_solution", "")
                    elif comparison_type == "llm_vs_corrupted":
                        ref = task_data.get("corrupted_solution", "")
                        pred = task_data.get("solution", "")
                    elif comparison_type == "llm_vs_canonical":
                        ref = task_data.get("canonical_solution", "")
                        pred = task_data.get("solution", "")

                    if ref and pred:
                        references.append(ref)
                        predictions.append(pred)

            if references and predictions:
                # Initialize comparison_type in avg_similarity if it doesn't exist
                if comparison_type not in avg_similarity:
                    avg_similarity[comparison_type] = {}

                # For corrupted_vs_canonical, only calculate with comments
                if comparison_type == "corrupted_vs_canonical":
                    avg_similarity[comparison_type]["corpus_bleu_score"] = calculate_corpus_bleu_score(references, predictions, ignore_comments=False)
                    avg_similarity[comparison_type]["corpus_codebleu"] = calculate_corpus_codebleu_score(references, predictions, ignore_comments=False)
                else:
                    # For LLM comparisons, calculate both with and without comments
                    avg_similarity[comparison_type]["corpus_bleu_score"] = calculate_corpus_bleu_score(references, predictions, ignore_comments=False)
                    avg_similarity[comparison_type]["corpus_codebleu"] = calculate_corpus_codebleu_score(references, predictions, ignore_comments=False)
                    avg_similarity[comparison_type]["corpus_no_comments_bleu_score"] = calculate_corpus_bleu_score(references, predictions, ignore_comments=True)
                    avg_similarity[comparison_type]["corpus_no_comments_codebleu"] = calculate_corpus_codebleu_score(references, predictions, ignore_comments=True)

    # Calculate pass@1
    pass_at_1 = passed_tasks / total_tasks if total_tasks > 0 else 0.0

    # Metadata
    metrics = {
        "pass@1": pass_at_1,
        "total_problems": len(problems),
        "evaluated_problems": total_tasks,
        "passed_problems": passed_tasks,
        "eval_similarity": eval_similarity,
        "similarity_metrics": avg_similarity if eval_similarity else None,
    }
    # Add metrics to results
    results["metrics"] = metrics

    # Save results
    if save_results:
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {result_path}")
    else:
        # Print results
        print(f"pass@1: {pass_at_1:.3f}")
        print(f"Evaluated {total_tasks}/{len(problems)} problems")
        print(f"Passed {passed_tasks}/{total_tasks} problems")

        if eval_similarity:
            similarity_count = sum(1 for task_data in results["eval"].values() if task_data.get("similarity_metrics"))
            print(f"Similarity metrics computed for {similarity_count} problems")

            if avg_similarity:
                print("\nSimilarity Metrics (Average and Corpus-Level):")
                for comparison_type, metrics_dict in avg_similarity.items():
                    print(f"  {comparison_type}:")

                    # Separate corpus metrics from average metrics
                    corpus_metrics = {}
                    average_metrics = {}

                    for metric_name, value in metrics_dict.items():
                        if metric_name.startswith("corpus_"):
                            corpus_metrics[metric_name] = value
                        else:
                            average_metrics[metric_name] = value

                    # Print average metrics first
                    if average_metrics:
                        # For corrupted_vs_canonical, we only have with-comments metrics
                        if comparison_type == "corrupted_vs_canonical":
                            print(f"    Average Metrics (With Comments Only):")
                            for metric_name, avg_value in average_metrics.items():
                                print(f"      {metric_name}: {avg_value:.4f}")
                        else:
                            # For LLM comparisons, separate by with/without comments
                            with_comments_metrics = {}
                            no_comments_metrics = {}

                            for metric_name, avg_value in average_metrics.items():
                                if metric_name.startswith("no_comments_"):
                                    no_comments_metrics[metric_name.replace("no_comments_", "")] = avg_value
                                elif not metric_name.startswith("body_no_comments_"):
                                    with_comments_metrics[metric_name] = avg_value

                            if with_comments_metrics:
                                print(f"    Average Metrics - With Comments:")
                                for metric_name, avg_value in with_comments_metrics.items():
                                    print(f"      {metric_name}: {avg_value:.4f}")

                            if no_comments_metrics:
                                print(f"    Average Metrics - Without Comments:")
                                for metric_name, avg_value in no_comments_metrics.items():
                                    print(f"      {metric_name}: {avg_value:.4f}")

                            body_with_comments = {k: v for k, v in average_metrics.items() if k.startswith("body_") and not k.startswith("body_no_comments_")}
                            if body_with_comments:
                                print(f"    Average Body Metrics (With Comments):")
                                for metric_name, avg_value in body_with_comments.items():
                                    print(f"      {metric_name}: {avg_value:.4f}")

                            body_no_comments = {k.replace("body_no_comments_", ""): v for k, v in average_metrics.items() if k.startswith("body_no_comments_")}
                            if body_no_comments:
                                print(f"    Average Body Metrics (Without Comments):")
                                for metric_name, avg_value in body_no_comments.items():
                                    print(f"      body_{metric_name}: {avg_value:.4f}")

                    # Print corpus metrics
                    if corpus_metrics:
                        print(f"    Corpus-Level Metrics:")
                        for metric_name, value in corpus_metrics.items():
                            print(f"      {metric_name}: {value:.4f}")

    return results, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simplified BigCodeBench Evaluator")
    parser.add_argument("--sample_path", help="Path to JSONL file containing solutions")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of worker processes (default: number of CPU cores)")
    parser.add_argument("--eval_similarity", action="store_true", default=False, help="Evaluate similarity metrics (BLEU, Levenshtein, edit distance)")
    parser.add_argument("--hard", action="store_true", default=False, help="Use bigcodebench-hard dataset instead of regular bigcodebench")

    args = parser.parse_args()

    try:
        results, metrics = evaluate(sample_path=args.sample_path, save_results=not args.no_save, max_workers=args.max_workers, eval_similarity=args.eval_similarity, hard=args.hard)
        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
