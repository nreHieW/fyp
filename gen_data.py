import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from partial_edits.utils.extract_utils import *

load_dotenv()

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

Guidelines for selecting bugs:
- Focus on logic/algorithmic errors rather than syntax errors
- Consider edge cases where the bug might manifest
- The bug should be discoverable through testing
- Avoid obvious mistakes like incorrect variable names or syntax errors

First, think step by step and come up with an approach to generate the corrupted solution.
Then, output the corrupted solution in a code block, maintaining exact formatting and whitespace of the original code except for your specific changes."""


def main():
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    bigcodebench_dataset = load_dataset("bigcode/bigcodebench-hard", cache_dir="data", split="v0.1.0_hf")

    # Output single JSONL file
    output_filepath = "corrupted_solutions.jsonl"

    generated_count = 0

    with open(output_filepath, "w") as output_file:
        for coding_problem in tqdm(bigcodebench_dataset):
            problem_description = coding_problem["instruct_prompt"]
            canonical_solution_body = coding_problem["canonical_solution"]  # only the function body
            function_signature_with_docstring = coding_problem["complete_prompt"]
            complete_canonical_solution = function_signature_with_docstring + "\n" + canonical_solution_body
            test_cases = coding_problem["test"]
            problem_id = coding_problem["task_id"]

            # original_test_result = safe_exec(complete_canonical_solution, test_cases)
            # assert original_test_result["is_correct"], f"Original solution for problem {problem_id} unexpectedly fails all tests!"

            extracted_docstring = extract_docstring(coding_problem["complete_prompt"])

            llm_completion = openai_client.chat.completions.create(
                model="deepseek/deepseek-r1-0528:free",
                messages=[
                    {"role": "system", "content": CORRUPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Problem Statement:\n{problem_description}\n\nCorrect Solution:\n{complete_canonical_solution}\n\nTest Code:\n{test_cases}"},
                ],
                temperature=0.0,
            )
            raw_llm_response = llm_completion.choices[0].message.content

            corrupted_solution_code = extract_code_from_response(raw_llm_response)
            if not extract_docstring(corrupted_solution_code):
                corrupted_solution_with_docstring = insert_docstring(corrupted_solution_code, extracted_docstring)
            else:
                corrupted_solution_with_docstring = corrupted_solution_code

            # Single entry with all data (compatible with evaluate.py)
            jsonl_entry = {
                "task_id": problem_id,
                "corrupted_solution": corrupted_solution_with_docstring,
                "prompt": problem_description,
                "canonical_solution": complete_canonical_solution,
                "test_code": test_cases,
                "llm_response": raw_llm_response,
            }
            output_file.write(json.dumps(jsonl_entry) + "\n")
            generated_count += 1

    print(f"Generated {generated_count} corrupted solutions")
    print(f"Output file: {output_filepath}")


if __name__ == "__main__":
    main()
