import json
import os
import argparse

import time
from datasets import load_dataset
from dotenv import load_dotenv
import requests
from tqdm import tqdm

from utils.extract_utils import *
from utils.code_corruptor import CodeCorruptor

load_dotenv(override=True)

EVALUATOR_URL = "http://localhost:8000"


def evaluate_solution(problem, solution, identifier, expect_fail=False, max_retries=3, retry_delay=5):
    """Evaluate a solution and return whether it meets expectations"""
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{EVALUATOR_URL}/evaluate", json={"completion_id": 0, "problem": problem, "solution": solution, "identifier": identifier})
            result = response.json()
            status = result["status"]
            details = result.get("details", {})
            if expect_fail:
                if status == "fail":
                    test_stats = details.get("TEST_STATS", {})
                    tests_run = test_stats.get("tests_run", 0)
                    failures = test_stats.get("failures", 0)
                    errors = test_stats.get("errors", 0)

                    if tests_run > 0:
                        total_failed = failures + errors
                        if total_failed == tests_run:
                            return True
                        else:
                            print(f"Partial failure for {identifier}: {total_failed}/{tests_run} tests failed")
                            return False
                    else:
                        return True
                else:
                    return False
            else:
                return status == "pass"

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Evaluation failed for {identifier} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Error evaluating {identifier} after {max_retries} attempts: {e}")
                return False


def load_existing_task_ids(output_filepath):
    """Load task_ids that have already been processed from existing output file."""
    processed_task_ids = set()
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        processed_task_ids.add(entry["task_id"])
        except Exception as e:
            print(f"Warning: Could not read existing file {output_filepath}: {e}")
            print("Starting fresh...")
            processed_task_ids = set()
    return processed_task_ids


def load_failed_canonical_cache(cache_filepath):
    """Load task_ids that have failed canonical solution checks."""
    failed_canonical_ids = set()
    if os.path.exists(cache_filepath):
        try:
            with open(cache_filepath, "r") as f:
                for line in f:
                    if line.strip():
                        failed_canonical_ids.add(line.strip())
        except Exception as e:
            print(f"Warning: Could not read failed canonical cache {cache_filepath}: {e}")
    return failed_canonical_ids


def save_failed_canonical(cache_filepath, task_id):
    """Save a task_id that failed canonical solution check."""
    with open(cache_filepath, "a") as f:
        f.write(task_id + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard", action="store_true", default=False, help="Use bigcodebench-hard dataset instead of regular bigcodebench")
    parser.add_argument("--n", type=int, default=400, help="Number of items to process (default: 400)")
    parser.add_argument("--check_canonical", action="store_true", default=False, help="Check that canonical solution passes all test cases before corrupting")
    parser.add_argument("--mutation_seed", type=int, default=42, help="Seed for mutation randomness (default: 42)")
    args = parser.parse_args()

    corruptor = CodeCorruptor(seed=args.mutation_seed)

    if args.hard:
        dataset_name = "bigcode/bigcodebench-hard"
    else:
        dataset_name = "bigcode/bigcodebench"

    bigcodebench_dataset = load_dataset(dataset_name, cache_dir="data", split="v0.1.0_hf")

    dataset_size = len(bigcodebench_dataset)

    print(f"Using dataset: {dataset_name}")
    print(f"Dataset size: {dataset_size}")

    os.makedirs("data/corrupted", exist_ok=True)
    os.makedirs("data/questions", exist_ok=True)
    output_filepath = f"data/questions/corrupted_solutions_manual_{'hard' if args.hard else 'easy'}_{args.n}.jsonl"
    failed_canonical_cache = f"data/corrupted/failed_canonical_cache_{'hard' if args.hard else 'easy'}.txt"

    # Load existing processed task_ids
    processed_task_ids = load_existing_task_ids(output_filepath)
    failed_canonical_ids = load_failed_canonical_cache(failed_canonical_cache) if args.check_canonical else set()
    print(f"Found {len(processed_task_ids)} already processed items")
    if args.check_canonical:
        print(f"Found {len(failed_canonical_ids)} cached failed canonical solutions")

    generated_count = 0
    failed_corruption_count = 0
    skipped_canonical_failures = 0
    dataset_index = 0
    data = []
    with open("/Users/weihern/Documents/NUS/fyp/data/questions/corrupted_solutions_manual_easy_400.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    ids = set([x["task_id"] for x in data])
    len(ids)

    with open(output_filepath, "a") as output_file:
        # Continue until we have generated exactly N successful mutations
        pbar = tqdm(total=args.n, desc="Generating corrupted solutions")
        pbar.update(len(processed_task_ids))  # Update progress with already completed items

        while generated_count < (args.n - len(processed_task_ids)) and dataset_index < dataset_size:
            coding_problem = bigcodebench_dataset[dataset_index]
            dataset_index += 1

            # Skip if already processed
            if coding_problem["task_id"] in processed_task_ids:
                continue

            # Skip if known to fail canonical check
            if args.check_canonical and coding_problem["task_id"] in failed_canonical_ids:
                skipped_canonical_failures += 1
                continue

            # Check canonical solution if flag is set
            if args.check_canonical:
                canonical_solution_body = coding_problem["canonical_solution"]
                function_signature_with_docstring = coding_problem["complete_prompt"]
                complete_canonical_solution = function_signature_with_docstring + "\n" + canonical_solution_body

                if not evaluate_solution(coding_problem, complete_canonical_solution, f"canonical_{coding_problem['task_id']}"):
                    print(f"Skipping {coding_problem['task_id']} - canonical solution fails test cases")
                    save_failed_canonical(failed_canonical_cache, coding_problem["task_id"])
                    failed_canonical_ids.add(coding_problem["task_id"])
                    skipped_canonical_failures += 1
                    continue

            problem_description = coding_problem["instruct_prompt"]
            canonical_solution_body = coding_problem["canonical_solution"]
            function_signature_with_docstring = coding_problem["complete_prompt"]
            complete_canonical_solution = function_signature_with_docstring + "\n" + canonical_solution_body
            test_cases = coding_problem["test"]
            problem_id = coding_problem["task_id"]
            extracted_docstring = extract_docstring(coding_problem["complete_prompt"])

            # Apply corruption
            corrupted_code, mutation_types = corruptor.corrupt_function(complete_canonical_solution, problem_id)

            if corrupted_code is None:
                print(f"Failed to corrupt {problem_id}: {mutation_types}")
                failed_corruption_count += 1
                continue

            # Ensure docstring is preserved
            if not extract_docstring(corrupted_code):
                corrupted_solution_with_docstring = insert_docstring(corrupted_code, extracted_docstring)
            else:
                corrupted_solution_with_docstring = corrupted_code

            # Verify that the corrupted solution actually fails the tests
            if not evaluate_solution(coding_problem, corrupted_solution_with_docstring, f"corrupted_{problem_id}", expect_fail=True):
                print(f"Warning: Corrupted solution for {problem_id} still passes tests, skipping...")
                failed_corruption_count += 1
                continue

            jsonl_entry = {
                "task_id": problem_id,
                "corrupted_solution": corrupted_solution_with_docstring,
                "prompt": problem_description,
                "canonical_solution": complete_canonical_solution,
                "test_code": test_cases,
                "mutation_types": mutation_types,
            }
            output_file.write(json.dumps(jsonl_entry) + "\n")
            output_file.flush()
            generated_count += 1
            pbar.update(1)

        pbar.close()

    total_processed = len(processed_task_ids) + generated_count
    print(f"Generated {generated_count} new corrupted solutions")
    print(f"Failed to corrupt {failed_corruption_count} items")
    if args.check_canonical:
        print(f"Skipped due to canonical solution failures: {skipped_canonical_failures}")
    print(f"Total processed items: {total_processed}")
    print(f"Output file: {output_filepath}")

    if total_processed < args.n:
        print(f"Warning: Only generated {total_processed} out of {args.n} requested items (dataset exhausted)")


if __name__ == "__main__":
    main()
