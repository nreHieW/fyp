import json
import os
import sys
import argparse
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
from datasets import load_dataset

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from icl_code.utils.style_compliance import load_judge_constraints, create_style_judge_system_prompt, create_style_judge_user_prompt, icl_style_compliance

# BigCodeBench evaluator constants
PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"
EVALUATOR_URL = "http://localhost:8000"


def load_solutions(sample_path: str) -> List[Dict]:
    """
    Load solutions from JSONL file.

    Expected JSONL format:
    - task_id: BigCodeBench task identifier
    - solution: Generated code solution
    - problem: Problem description
    - num_shots: Number of few-shot examples used
    - Other optional fields: reasoning, canonical_solution, test_code, etc.
    """
    solutions = []

    with open(sample_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                sample = json.loads(line.strip())
                if "solution" not in sample:
                    print(f"Warning: No 'solution' field in line {i+1}, skipping")
                    continue
                sample["_identifier"] = f"{sample['task_id']} (line {i+1} in {sample_path})"
                solutions.append(sample)

    return solutions


def get_bigcodebench(hard: bool = False) -> Dict[str, Any]:

    if hard:
        dataset_name = "bigcode/bigcodebench-hard"
    else:
        dataset_name = "bigcode/bigcodebench"

    dataset = load_dataset(dataset_name, split="v0.1.4", cache_dir="data")
    return {item["task_id"]: item for item in dataset}


def check_correctness(task_info: Tuple[int, Dict[str, Any], str, str]) -> Dict:
    """Check functional correctness using BigCodeBench evaluator."""
    completion_id, problem, solution, identifier = task_info

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{EVALUATOR_URL}/evaluate", json={"completion_id": completion_id, "problem": problem, "solution": solution, "identifier": identifier})
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 10
                print(f"Evaluation failed for {identifier}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e


def evaluate_style_compliance(solutions: List[Dict], judge_model_name: str, judge_constraints: str, judge_is_reasoning: bool = False) -> Dict[str, Dict]:
    """
    Evaluate style compliance using LLM judge.

    Args:
        solutions: List of solution dictionaries
        judge_model_name: Model to use as judge
        judge_constraints: Judge constraints text for evaluation
        judge_is_reasoning: Whether judge model uses reasoning

    Returns:
        Dict mapping task_id to style compliance results
    """
    print(f"Evaluating style compliance using {judge_model_name} as judge...")

    # Initialize judge model
    judge_model = get_model(judge_model_name, judge_is_reasoning)

    system_prompt = create_style_judge_system_prompt(judge_constraints)

    # Prepare batch requests
    user_messages = []
    task_ids = []

    for solution in solutions:
        task_id = solution["task_id"]
        code = solution["solution"]

        user_message = create_style_judge_user_prompt(code)
        user_messages.append(user_message)
        task_ids.append(task_id)

    # Generate style evaluations
    print(f"Generating style evaluations for {len(user_messages)} solutions...")
    style_results = {}

    try:
        batch_responses = judge_model.generate_responses(system_prompt, user_messages)

        for task_id, response in zip(task_ids, batch_responses):
            if response.get("error"):
                print(f"Error in style evaluation for {task_id}: {response['error']} - skipping this sample")
                continue
            else:
                llm_response = response.get("final_answer", "")
                result = icl_style_compliance(llm_response)

            style_results[task_id] = {
                "compliance_score": result["compliance_score"],
                "rules_passed": result["rules_passed"],
                "judge_response": response.get("final_answer", ""),
                "judge_reasoning": response.get("reasoning", ""),
            }

    except Exception as e:
        print(f"Error in style evaluation: {e} - continuing without style evaluation for this batch")
        # No fallback - errored samples will be retried on next run
        pass

    print(f"Style evaluation completed for {len(style_results)} solutions")
    return style_results


def evaluate(
    sample_path: str,
    judge_constraints_path: str = None,
    judge_model: str = None,
    judge_is_reasoning: bool = False,
    save_results: bool = True,
    max_workers: int = 8,
    hard: bool = False,
    eval_style: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Evaluate solutions for both functional correctness and style compliance.

    Args:
        sample_path: Path to JSONL file containing solutions
        judge_constraints_path: Path to judge constraints file
        judge_model: Model to use for style evaluation (if None, use programmatic checks only)
        judge_is_reasoning: Whether judge model uses reasoning
        save_results: Whether to save results to file
        max_workers: Number of worker processes for functional evaluation
        hard: Whether solutions were generated for BigCodeBench-Hard
        eval_style: Whether to evaluate style compliance

    Returns:
        Tuple of (results_dict, metrics_dict)
    """
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

    print("Loading solutions...")
    solutions = load_solutions(sample_path)
    print(f"Loaded {len(solutions)} solutions")

    # Check for existing results to resume evaluation
    processed_task_ids = set()
    existing_eval_results = {}
    if save_results and os.path.exists(result_path):
        try:
            with open(result_path, "r") as f:
                existing_data = json.load(f)
                existing_eval_results = existing_data.get("eval", {})
                # Only consider successfully evaluated samples (no evaluation_error)
                for task_id, result in existing_eval_results.items():
                    if not result.get("evaluation_error", False):
                        processed_task_ids.add(task_id)
                print(f"Found existing results for {len(processed_task_ids)} successfully evaluated samples")
                if len(existing_eval_results) > len(processed_task_ids):
                    print(f"Found {len(existing_eval_results) - len(processed_task_ids)} samples with evaluation errors - will retry")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    # Filter solutions that have corresponding problems
    valid_solutions = []
    for solution in solutions:
        if solution["task_id"] in problems:
            valid_solutions.append(solution)
        else:
            print(f"Warning: No problem found for task_id {solution['task_id']}")

    # Filter out already processed solutions
    solutions_to_evaluate = [s for s in valid_solutions if s["task_id"] not in processed_task_ids]
    print(f"Found {len(valid_solutions)} valid solutions, {len(solutions_to_evaluate)} need evaluation")

    if not solutions_to_evaluate:
        if processed_task_ids:
            print("All solutions already evaluated successfully.")
            # Return existing results
            with open(result_path, "r") as f:
                existing_data = json.load(f)
            return existing_data, existing_data.get("metrics", {})
        else:
            print("No valid solutions found to evaluate.")
            return {}, {"pass@1": 0.0, "style_compliance@1": 0.0, "both_correct@1": 0.0}

    solutions = solutions_to_evaluate

    # Evaluate functional correctness
    print(f"Evaluating functional correctness using {max_workers} workers...")

    tasks = []
    completion_id = defaultdict(int)

    for sample in solutions:
        task_id = sample["task_id"]
        solution = sample["solution"]

        if task_id in problems:
            task_info = (completion_id[task_id], problems[task_id], solution, sample["_identifier"])
            tasks.append(task_info)
            completion_id[task_id] += 1

    functional_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(check_correctness, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Functional evaluation") as pbar:
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    task_id = result["task_id"]
                    functional_results[task_id] = result
                except Exception as e:
                    task_info = future_to_task[future]
                    print(f"Error evaluating task {task_info[1]['task_id']}: {e}")

                pbar.update(1)

    # Load judge constraints if needed
    judge_constraints = ""
    if eval_style and judge_constraints_path:
        judge_constraints = load_judge_constraints(judge_constraints_path)
        print(f"Loaded judge constraints from {judge_constraints_path}")

    # Evaluate style compliance
    style_results = {}
    if eval_style:
        if judge_model and judge_constraints:
            style_results = evaluate_style_compliance(solutions, judge_model, judge_constraints, judge_is_reasoning)
        else:
            if not judge_model:
                print("No judge model specified, skipping style evaluation")
            if not judge_constraints:
                print("No judge constraints provided, skipping style evaluation")

    # Combine results and calculate metrics
    combined_results = {}
    passed_functional = 0
    passed_style = 0
    passed_both = 0
    total_evaluated = 0

    for solution in solutions:
        task_id = solution["task_id"]

        # Get functional result
        func_result = functional_results.get(task_id, {})
        is_functional_correct = func_result.get("status") == PASS

        # Get style result - if no style result, mark as error for retry
        style_result = style_results.get(task_id, {})
        if not style_result and eval_style and judge_model:
            # Style evaluation failed - mark for retry
            print(f"Style evaluation failed for {task_id} - will retry on next run")
            style_result = {"compliance": {}, "compliance_score": 0.0, "rules_passed": 0, "total_rules": 0, "judge_response": "", "judge_reasoning": "", "evaluation_error": True}

        is_style_compliant = style_result.get("compliance_score", 0.0) >= 0.5  # At least 50% of rules

        # Combine results
        combined_results[task_id] = {
            "task_id": task_id,
            "solution": solution["solution"],
            "problem": solution.get("problem", ""),
            "num_shots": solution.get("num_shots", 0),
            "generation_model": solution.get("generation_model", ""),
            "is_reasoning": solution.get("is_reasoning", False),
            # Functional correctness
            "functional_status": func_result.get("status", "error"),
            "functional_details": func_result.get("details", ""),
            "is_functional_correct": is_functional_correct,
            # Style compliance
            "style_compliance": style_result.get("compliance", {}),
            "style_compliance_score": style_result.get("compliance_score", 0.0),
            "style_rules_passed": style_result.get("rules_passed", 0),
            "style_total_rules": style_result.get("total_rules", 0),
            "is_style_compliant": is_style_compliant,
            "judge_response": style_result.get("judge_response", ""),
            "evaluation_error": style_result.get("evaluation_error", False),
            # Combined
            "is_both_correct": is_functional_correct and is_style_compliant,
            # Additional fields
            "reasoning": solution.get("reasoning", ""),
            "canonical_solution": solution.get("canonical_solution", ""),
            "test_code": solution.get("test_code", ""),
        }

        # Update counters only if not an evaluation error
        if not style_result.get("evaluation_error", False):
            if is_functional_correct:
                passed_functional += 1
            if is_style_compliant:
                passed_style += 1
            if is_functional_correct and is_style_compliant:
                passed_both += 1
            total_evaluated += 1

    # Combine with existing results
    all_combined_results = existing_eval_results.copy()
    all_combined_results.update(combined_results)

    # Recalculate metrics including existing results
    total_all = 0
    passed_all_functional = 0
    passed_all_style = 0
    passed_all_both = 0
    total_style_score = 0.0

    for result in all_combined_results.values():
        if not result.get("evaluation_error", False):
            total_all += 1
            if result.get("is_functional_correct", False):
                passed_all_functional += 1
            if result.get("is_style_compliant", False):
                passed_all_style += 1
            if result.get("is_both_correct", False):
                passed_all_both += 1
            total_style_score += result.get("style_compliance_score", 0.0)

    # Calculate metrics
    pass_at_1 = passed_all_functional / total_all if total_all > 0 else 0.0
    style_compliance_at_1 = passed_all_style / total_all if total_all > 0 else 0.0
    both_correct_at_1 = passed_all_both / total_all if total_all > 0 else 0.0
    avg_style_score = total_style_score / total_all if total_all > 0 else 0.0

    metrics = {
        "pass@1": pass_at_1,
        "style_compliance@1": style_compliance_at_1,
        "both_correct@1": both_correct_at_1,
        "avg_style_compliance_score": avg_style_score,
        "total_problems": len(problems),
        "evaluated_problems": total_all,
        "passed_functional": passed_all_functional,
        "passed_style": passed_all_style,
        "passed_both": passed_all_both,
        "evaluation_errors": len(all_combined_results) - total_all,
        "eval_style": eval_style,
        "judge_model": judge_model if judge_model else "programmatic",
    }

    # Prepare final results
    results = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "sample_path": sample_path, "dataset": dataset_name, "metrics": metrics, "eval": all_combined_results}

    # Save results
    if save_results:
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_path}")

    # Print summary
    print(f"\nEvaluation completed!")
    print(f"Functional correctness (pass@1): {pass_at_1:.3f} ({passed_all_functional}/{total_all})")
    if eval_style:
        print(f"Style compliance@1: {style_compliance_at_1:.3f} ({passed_all_style}/{total_all})")
        print(f"Both correct@1: {both_correct_at_1:.3f} ({passed_all_both}/{total_all})")
        print(f"Average style score: {avg_style_score:.3f}")
        if len(all_combined_results) > total_all:
            print(f"Evaluation errors: {len(all_combined_results) - total_all} samples need retry")

    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL code solutions for functional correctness and style compliance")
    parser.add_argument("--sample_path", required=True, help="Path to JSONL file containing solutions")
    parser.add_argument("--judge_constraints_path", default="data/icl_code/judge.txt", help="Path to judge constraints file")
    parser.add_argument("--judge_model", help="Model to use for style evaluation (if not specified, use programmatic checks)")
    parser.add_argument("--judge_is_reasoning", action="store_true", help="Whether judge model uses reasoning")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of worker processes for functional evaluation")
    parser.add_argument("--hard", action="store_true", help="Use BigCodeBench-Hard dataset")
    parser.add_argument("--no-style", action="store_true", help="Skip style compliance evaluation")

    args = parser.parse_args()

    if not os.path.exists(args.sample_path):
        print(f"Error: Sample file not found at {args.sample_path}")
        sys.exit(1)

    try:
        evaluate(
            sample_path=args.sample_path,
            judge_constraints_path=args.judge_constraints_path,
            judge_model=args.judge_model,
            judge_is_reasoning=args.judge_is_reasoning,
            save_results=not args.no_save,
            max_workers=args.max_workers,
            hard=args.hard,
            eval_style=not args.no_style,
        )

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
