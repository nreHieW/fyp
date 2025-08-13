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
    tokenize_code,
    get_codebleu_score,
    calculate_bleu_score,
    calculate_corpus_bleu_score,
    calculate_corpus_codebleu_score,
    get_diffsitter_edit_distance,
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


def _compute_raw_metrics(reference: str, prediction: str, ignore_comments: bool = False) -> Dict[str, float]:
    """Compute raw similarity metrics for two text strings"""
    metrics = {
        "bleu_score": calculate_bleu_score(reference, prediction, ignore_comments=ignore_comments),
        "levenshtein_distance": get_levenshtein_distance(reference, prediction, ignore_comments=ignore_comments),
        "edit_distance": count_diff_lines(reference, prediction, ignore_comments=ignore_comments),
        "diffsitter_edit_distance": get_diffsitter_edit_distance(reference, prediction),
        "meteor_score": get_meteor_score(reference, prediction, ignore_comments=ignore_comments),
        "chrf_score": get_chrf_score(reference, prediction, ignore_comments=ignore_comments),
    }

    rouge_scores = get_rouge_scores(reference, prediction, ignore_comments=ignore_comments)
    metrics.update(rouge_scores)

    codebleu_scores = get_codebleu_score(reference, prediction, ignore_comments=ignore_comments)
    metrics.update(codebleu_scores)

    return metrics


def _normalize_metrics(raw_metrics: Dict[str, float], reference: str, prediction: str, prefix: str = "", ignore_comments: bool = False) -> Dict[str, float]:
    """Normalize distance metrics and add prefix if provided"""
    if ignore_comments:
        reference = standardize_code_formatting(reference)
        prediction = standardize_code_formatting(prediction)

    max_lines = max(len(reference.splitlines()), len(prediction.splitlines()))
    max_tokens = max(len(tokenize_code(reference)), len(tokenize_code(prediction)))

    result = {}

    # Add all raw metrics with prefix
    for key, value in raw_metrics.items():
        result[f"{prefix}{key}"] = value

    result[f"{prefix}normalized_levenshtein"] = raw_metrics["levenshtein_distance"] / max_tokens if max_tokens > 0 else 0.0
    result[f"{prefix}normalized_edit_distance"] = raw_metrics["edit_distance"] / max_lines if max_lines > 0 else 0.0
    if raw_metrics["diffsitter_edit_distance"] == -1:
        result[f"{prefix}normalized_diffsitter_edit_distance"] = -1.0
    else:
        result[f"{prefix}normalized_diffsitter_edit_distance"] = raw_metrics["diffsitter_edit_distance"] / max_lines if max_lines > 0 else 0.0

    return result


class SimilarityCalculator:
    def __init__(self, get_body_cached=None):
        self.get_body_cached = get_body_cached
        self._zero_metrics = {
            "bleu_score": 0.0,
            "levenshtein_distance": 0.0,
            "edit_distance": 0.0,
            "diffsitter_edit_distance": 0.0,
            "meteor_score": 0.0,
            "chrf_score": 0.0,
            "rouge1_fmeasure": 0.0,
            "rouge2_fmeasure": 0.0,
            "rougeL_fmeasure": 0.0,
            "codebleu": 0.0,
        }

    def _extract_body(self, code: str) -> str:
        if self.get_body_cached:
            return self.get_body_cached(code)
        return extract_function_body(code)

    def calculate_metrics(self, reference: str, prediction: str, body_only=False, with_comments=True, no_comments=True, prefix=""):
        if body_only:
            reference = self._extract_body(reference)
            prediction = self._extract_body(prediction)
            if not reference or not prediction:
                zero_with = _normalize_metrics(self._zero_metrics, "", "", f"{prefix}body_", False) if with_comments else {}
                zero_no = _normalize_metrics(self._zero_metrics, "", "", f"{prefix}body_no_comments_", True) if no_comments else {}
                return {**zero_with, **zero_no}

        result = {}
        if with_comments:
            raw = _compute_raw_metrics(reference, prediction, False)
            body_prefix = f"{prefix}body_" if body_only else prefix
            result.update(_normalize_metrics(raw, reference, prediction, body_prefix, False))

        if no_comments:
            raw = _compute_raw_metrics(reference, prediction, True)
            body_prefix = f"{prefix}body_no_comments_" if body_only else f"{prefix}no_comments_"
            result.update(_normalize_metrics(raw, reference, prediction, body_prefix, True))

        return result


def load_solutions(sample_path: str):
    """
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


def _prepare_evaluation_tasks(sample_path: str, problems: Dict) -> Tuple[List[Tuple], List[Dict]]:
    solutions = load_solutions(sample_path)
    tasks = []
    completion_id = Counter()

    for sample in solutions:
        task_id = sample["task_id"]
        if task_id not in problems:
            continue
        task_info = (completion_id[task_id], problems[task_id], sample["solution"], sample["_identifier"])
        tasks.append(task_info)
        completion_id[task_id] += 1

    return tasks, solutions


def _run_evaluations(tasks: List[Tuple], max_workers: int) -> List[Dict]:
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(check_correctness, task): task for task in tasks}
        with tqdm(total=len(tasks), desc="Evaluating") as pbar:
            for future in as_completed(future_to_task):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    task_info = future_to_task[future]
                    print(f"Error evaluating task {task_info[1]['task_id']}: {e}")
                pbar.update(1)
    return all_results


def _process_similarity_metrics(task_data: Dict, eval_similarity: bool, similarity_calc: SimilarityCalculator) -> Dict:
    if not eval_similarity:
        return {}

    canonical = task_data.get("canonical_solution", "")
    corrupted = task_data.get("corrupted_solution", "")
    llm = task_data.get("solution", "")

    metrics = {}
    if corrupted and canonical:
        # canonical_vs_corrupted: prediction=canonical, reference=corrupted
        metrics["canonical_vs_corrupted"] = similarity_calc.calculate_metrics(corrupted, canonical)
        metrics["canonical_vs_corrupted"].update(similarity_calc.calculate_metrics(corrupted, canonical, body_only=True))

    if llm and corrupted:
        # llm_vs_corrupted: prediction=llm, reference=corrupted
        metrics["llm_vs_corrupted"] = similarity_calc.calculate_metrics(corrupted, llm)
        metrics["llm_vs_corrupted"].update(similarity_calc.calculate_metrics(corrupted, llm, body_only=True))

    if llm and canonical:
        # llm_vs_canonical: prediction=llm, reference=canonical
        metrics["llm_vs_canonical"] = similarity_calc.calculate_metrics(canonical, llm)
        metrics["llm_vs_canonical"].update(similarity_calc.calculate_metrics(canonical, llm, body_only=True))

    return metrics


def _calculate_corpus_metrics(results: Dict, eval_similarity: bool) -> Dict:
    if not eval_similarity:
        return {}

    avg_similarity = {}
    metric_sums, metric_counts = {}, {}

    for task_data in results["eval"].values():
        if not task_data.get("similarity_metrics"):
            continue
        for comparison_type, metrics in task_data["similarity_metrics"].items():
            if comparison_type not in metric_sums:
                metric_sums[comparison_type] = {}
                metric_counts[comparison_type] = {}
            for metric_name, value in metrics.items():
                metric_sums[comparison_type].setdefault(metric_name, 0)
                metric_counts[comparison_type].setdefault(metric_name, 0)
                metric_sums[comparison_type][metric_name] += value
                metric_counts[comparison_type][metric_name] += 1

    for comparison_type in metric_sums:
        avg_similarity[comparison_type] = {}
        for metric_name in metric_sums[comparison_type]:
            avg_similarity[comparison_type][metric_name] = metric_sums[comparison_type][metric_name] / metric_counts[comparison_type][metric_name]

    for comparison_type in ["canonical_vs_corrupted", "llm_vs_corrupted", "llm_vs_canonical"]:
        references, predictions = [], []
        for task_data in results["eval"].values():
            if not (task_data.get("similarity_metrics") and comparison_type in task_data["similarity_metrics"]):
                continue
            if comparison_type == "canonical_vs_corrupted":
                # canonical_vs_corrupted: prediction=canonical, reference=corrupted
                ref, pred = task_data.get("corrupted_solution", ""), task_data.get("canonical_solution", "")
            elif comparison_type == "llm_vs_corrupted":
                # llm_vs_corrupted: prediction=llm, reference=corrupted
                ref, pred = task_data.get("corrupted_solution", ""), task_data.get("solution", "")
            else:
                # llm_vs_canonical: prediction=llm, reference=canonical
                ref, pred = task_data.get("canonical_solution", ""), task_data.get("solution", "")
            if ref and pred:
                references.append(ref)
                predictions.append(pred)

        if references and predictions:
            avg_similarity.setdefault(comparison_type, {})
            if comparison_type == "canonical_vs_corrupted":
                avg_similarity[comparison_type]["corpus_bleu_score"] = calculate_corpus_bleu_score(references, predictions, False)
                avg_similarity[comparison_type]["corpus_codebleu"] = calculate_corpus_codebleu_score(references, predictions, ignore_comments=False)
            else:
                avg_similarity[comparison_type]["corpus_bleu_score"] = calculate_corpus_bleu_score(references, predictions, False)
                avg_similarity[comparison_type]["corpus_codebleu"] = calculate_corpus_codebleu_score(references, predictions, ignore_comments=False)
                avg_similarity[comparison_type]["corpus_no_comments_bleu_score"] = calculate_corpus_bleu_score(references, predictions, True)
                avg_similarity[comparison_type]["corpus_no_comments_codebleu"] = calculate_corpus_codebleu_score(references, predictions, ignore_comments=True)

    return avg_similarity


def _compute_difference_with_canonical(similarity_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute per-metric differences: canonical_vs_corrupted - llm_vs_corrupted.

    Only overlapping metric keys are considered. Missing keys are ignored.
    """
    if not similarity_metrics:
        return {}
    canonical_metrics = similarity_metrics.get("canonical_vs_corrupted", {})
    llm_vs_corrupted_metrics = similarity_metrics.get("llm_vs_corrupted", {})
    if not canonical_metrics or not llm_vs_corrupted_metrics:
        return {}
    diff: Dict[str, float] = {}
    for metric_name, canonical_value in canonical_metrics.items():
        if metric_name in llm_vs_corrupted_metrics:
            diff[metric_name] = canonical_value - llm_vs_corrupted_metrics[metric_name]
    return diff


def evaluate(sample_path: str, save_results: bool = True, max_workers: int = 8, eval_similarity: bool = False, hard: bool = False):
    assert sample_path and sample_path.endswith(".jsonl"), "Sample path must be a JSONL file"

    result_path = os.path.join(os.path.dirname(sample_path), "results", os.path.basename(sample_path).replace(".jsonl", "_eval_results.json"))
    if save_results:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

    dataset_name = "bigcode/bigcodebench-hard" if hard else "bigcode/bigcodebench"
    print(f"Loading BigCodeBench dataset: {dataset_name}...")
    problems = get_bigcodebench(hard=hard)

    print("Preparing solutions for evaluation...")
    tasks, solutions = _prepare_evaluation_tasks(sample_path, problems)

    if not tasks:
        print("No valid tasks found to evaluate.")
        return {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "eval": {}, "metrics": {}}, {"pass@1": 0.0, "total_problems": len(problems), "evaluated_problems": 0, "passed_problems": 0}

    print(f"Evaluating {len(tasks)} solutions using {max_workers} workers...")
    all_results = _run_evaluations(tasks, max_workers)

    print("Processing results and calculating diffs...")
    results = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "eval": {}, "metrics": {}}

    bodies_cache = {}

    def get_body_cached(solution: str) -> str:
        if solution not in bodies_cache:
            bodies_cache[solution] = extract_function_body(solution)
        return bodies_cache[solution]

    similarity_calc = SimilarityCalculator(get_body_cached)
    passed_tasks = 0

    for result in all_results:
        task_id = result["task_id"]
        original_sample = next((s for s in solutions if s["task_id"] == task_id), None)

        task_data = {
            "task_id": task_id,
            "solution": result["solution"],
            "canonical_solution": original_sample.get("canonical_solution") if original_sample else None,
            "corrupted_solution": original_sample.get("corrupted_solution") if original_sample else None,
            "status": result["status"],
            "details": result["details"],
            "diffs": calculate_diffs(
                original_sample.get("corrupted_solution", "") if original_sample else "", original_sample.get("canonical_solution", "") if original_sample else "", result["solution"]
            ),
            "similarity_metrics": _process_similarity_metrics(
                {
                    "canonical_solution": original_sample.get("canonical_solution") if original_sample else None,
                    "corrupted_solution": original_sample.get("corrupted_solution") if original_sample else None,
                    "solution": result["solution"],
                },
                eval_similarity,
                similarity_calc,
            ),
            "prompt": original_sample.get("prompt", "") if original_sample else "",
            "llm_response": original_sample.get("llm_response", "") if original_sample else "",
            "token_usage": original_sample.get("token_usage", {}) if original_sample else {},
            "llm_reasoning": original_sample.get("llm_reasoning", "") if original_sample else "",
        }

        # Compute and store the difference only inside similarity_metrics for averaging
        task_data["similarity_metrics"]["difference_with_canonical"] = _compute_difference_with_canonical(task_data.get("similarity_metrics", {}))
        results["eval"][task_id] = task_data
        if result["status"] == PASS:
            passed_tasks += 1

    avg_similarity = _calculate_corpus_metrics(results, eval_similarity)

    metrics = {
        "pass@1": passed_tasks / len(all_results) if all_results else 0.0,
        "total_problems": len(problems),
        "evaluated_problems": len(all_results),
        "passed_problems": passed_tasks,
        "eval_similarity": eval_similarity,
        "similarity_metrics": avg_similarity if eval_similarity else None,
    }
    results["metrics"] = metrics

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
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of worker processes (default: number of CPU cores)")
    parser.add_argument("--eval_similarity", action="store_true", default=False, help="Evaluate similarity metrics (BLEU, Levenshtein, edit distance)")
    parser.add_argument("--hard", action="store_true", default=False, help="Use bigcodebench-hard dataset instead of regular bigcodebench")

    args = parser.parse_args()

    results, metrics = evaluate(sample_path=args.sample_path, save_results=not args.no_save, max_workers=args.max_workers, eval_similarity=args.eval_similarity, hard=args.hard)
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
