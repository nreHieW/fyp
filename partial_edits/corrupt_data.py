import json
import os
import argparse
import sys
from datasets import load_dataset
from dotenv import load_dotenv
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.extract_utils import *
from models import get_model

load_dotenv(override=True)

EVALUATOR_URL = "http://localhost:8000"


def check_canonical_solution(problem, canonical_solution):
    """Check if the canonical solution passes all test cases"""
    try:
        response = requests.post(f"{EVALUATOR_URL}/evaluate", json={"completion_id": 0, "problem": problem, "solution": canonical_solution, "identifier": f"canonical_{problem['task_id']}"})
        result = response.json()
        return result["status"] == "pass"
    except Exception as e:
        print(f"Error checking canonical solution for {problem['task_id']}: {e}")
        return False


CORRUPTION_SYSTEM_PROMPT = """
You are a Python Expert specializing in code analysis and debugging. You will be provided with a problem statement, a correct solution, and the test cases. Your task is to create a subtly incorrect version of the solution such that the test cases fail:

Requirements for the corrupted solution:
1. Introduce exactly one logical bug that makes the solution incorrect
2. The bug should be subtle and not immediately obvious from reading the code
3. Prefer bugs that could realistically be made by a programmer (e.g., off-by-one errors, incorrect operators, swapped variables)
4. Changes must be minimal - modify as few lines of code as possible
5. Maintain the original function signature, default arguments, and overall structure
6. Preserve all existing comments and docstrings
7. The bug should not cause runtime errors or exceptions
8. Do not leave any comments or hints in the code about where the bug is that would help the user identify the bug

Guidelines for selecting bugs:
- Focus on logic/algorithmic errors rather than syntax errors
- Consider edge cases where the bug might manifest
- The bug should be discoverable through testing
- Avoid obvious mistakes like incorrect variable names or syntax errors
- Try to make the bug as subtle as possible
- Do not make changes to the error message or print statements
- When someone is provided with your corruption, they should not need to have access to the test cases to identify the bug

First, think step by step and come up with an approach to generate the corrupted solution.
Then, output the corrupted solution in a code block, maintaining exact formatting and whitespace of the original code except for your specific changes. Remember to try to limit the number of changes you make to the code that still causes the test cases to fail."""


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to use for corruption")
    parser.add_argument("--hard", action="store_true", default=False, help="Use bigcodebench-hard dataset instead of regular bigcodebench")
    parser.add_argument("--n", type=int, default=400, help="Number of items to process (default: 400)")
    parser.add_argument("--is_reasoning", action="store_true", default=False, help="Whether the model supports reasoning")
    parser.add_argument("--check_canonical", action="store_true", default=False, help="Check that canonical solution passes all test cases before corrupting")
    args = parser.parse_args()

    model = get_model(args.model, args.is_reasoning)

    if args.hard:
        dataset_name = "bigcode/bigcodebench-hard"
    else:
        dataset_name = "bigcode/bigcodebench"

    bigcodebench_dataset = load_dataset(dataset_name, cache_dir="data", split="v0.1.0_hf")

    dataset_size = len(bigcodebench_dataset)
    n_items = min(args.n, dataset_size)

    print(f"Using dataset: {dataset_name}")
    print(f"Dataset size: {dataset_size}")

    os.makedirs("data/corrupted", exist_ok=True)
    output_filepath = f"data/corrupted/corrupted_solutions_{args.model.replace('/', '_')}_{'hard' if args.hard else 'easy'}_{n_items}_{'reasoning' if args.is_reasoning else 'no_reasoning'}.jsonl"

    # Load existing processed task_ids
    processed_task_ids = load_existing_task_ids(output_filepath)
    print(f"Found {len(processed_task_ids)} already processed items")

    # Filter dataset to only include unprocessed items
    unprocessed_items = []
    skipped_canonical_failures = 0

    for item in bigcodebench_dataset:
        if item["task_id"] not in processed_task_ids:
            # Check canonical solution if flag is set
            if args.check_canonical:
                canonical_solution_body = item["canonical_solution"]
                function_signature_with_docstring = item["complete_prompt"]
                complete_canonical_solution = function_signature_with_docstring + "\n" + canonical_solution_body

                if not check_canonical_solution(item, complete_canonical_solution):
                    print(f"Skipping {item['task_id']} - canonical solution fails test cases")
                    skipped_canonical_failures += 1
                    continue

            unprocessed_items.append(item)

            # Stop when we have enough items to process
            if len(unprocessed_items) >= args.n:
                break

    print(f"Remaining items to process: {len(unprocessed_items)}")
    if args.check_canonical:
        print(f"Skipped due to canonical solution failures: {skipped_canonical_failures}")

    if len(unprocessed_items) == 0:
        print("All items have already been processed!")
        model.print_usage()
        return

    batch_data = []
    for coding_problem in unprocessed_items:
        problem_description = coding_problem["instruct_prompt"]
        canonical_solution_body = coding_problem["canonical_solution"]  # only the function body
        function_signature_with_docstring = coding_problem["complete_prompt"]
        complete_canonical_solution = function_signature_with_docstring + "\n" + canonical_solution_body
        test_cases = coding_problem["test"]
        problem_id = coding_problem["task_id"]
        extracted_docstring = extract_docstring(coding_problem["complete_prompt"])

        user_prompt = f"Problem Statement:\n{problem_description}\n\nCorrect Solution:\n{complete_canonical_solution}\n\nTest Code:\n{test_cases}"

        batch_data.append(
            {
                "user_prompt": user_prompt,
                "problem_id": problem_id,
                "problem_description": problem_description,
                "complete_canonical_solution": complete_canonical_solution,
                "test_cases": test_cases,
                "extracted_docstring": extracted_docstring,
            }
        )

    # Extract just the prompts for batch generation
    user_prompts = [item["user_prompt"] for item in batch_data]

    responses = model.generate_responses(CORRUPTION_SYSTEM_PROMPT, user_prompts)

    generated_count = 0

    with open(output_filepath, "a") as output_file:
        for i, (response, data) in enumerate(zip(responses, batch_data)):
            # Handle potential errors
            if "error" in response:
                print(f"Error generating corruption for {data['problem_id']}: {response['error']}")
                continue

            raw_llm_response = response["final_answer"]
            reasoning = response.get("reasoning", "")

            corrupted_solution_code = extract_code_from_response(raw_llm_response)
            corrupted_solution_code = standardize_code_formatting(corrupted_solution_code)
            if not extract_docstring(corrupted_solution_code):
                corrupted_solution_with_docstring = insert_docstring(corrupted_solution_code, data["extracted_docstring"])
            else:
                corrupted_solution_with_docstring = corrupted_solution_code

            jsonl_entry = {
                "task_id": data["problem_id"],
                "corrupted_solution": corrupted_solution_with_docstring,
                "prompt": data["problem_description"],
                "canonical_solution": data["complete_canonical_solution"],
                "test_code": data["test_cases"],
                "llm_response": raw_llm_response,
                "reasoning": reasoning,  # Include reasoning if available
            }
            output_file.write(json.dumps(jsonl_entry) + "\n")
            output_file.flush()  # Ensure data is written immediately
            generated_count += 1

    total_processed = len(processed_task_ids) + generated_count
    print(f"Generated {generated_count} new corrupted solutions")
    print(f"Total processed items: {total_processed}")
    print(f"Output file: {output_filepath}")

    model.print_usage()


if __name__ == "__main__":
    main()
