import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from partial_edits.utils.code_utils import safe_exec
from partial_edits.utils.extract_utils import *

load_dotenv()

SYSTEM_PROMPT = """
You are a Python Expert specializing in code analysis and debugging. You will be provided with a problem statement and a correct solution. Your task is to create a subtly incorrect version of the solution:

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
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    data = load_dataset("bigcode/bigcodebench-hard", cache_dir="data", split="v0.1.0_hf")
    corrupted_solutions = []

    for problem in tqdm(data):
        if len(corrupted_solutions) >= 10:
            break

        problem_statement = problem["instruct_prompt"]
        original_solution_body = problem["canonical_solution"]  # only the function body
        func_desc_with_docstring = problem["complete_prompt"]
        complete_original_solution = func_desc_with_docstring + "\n" + original_solution_body
        test_code = problem["test"]
        task_id = problem["task_id"]

        original_test_result = safe_exec(complete_original_solution, test_code)
        assert original_test_result["is_correct"], f"Original solution for problem {task_id} unexpectedly fails all tests!"

        docstring = extract_docstring(problem["complete_prompt"])

        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Problem Statement:\n{problem_statement}\n\nCorrect Solution:\n{original_solution_body}"}],
            temperature=0.0,
        )
        llm_response = completion.choices[0].message.content
        print(llm_response)

        corrupted_solution = extract_code_from_response(llm_response)
        corrupted_solution_with_docstring = insert_docstring(corrupted_solution, docstring)
        corrupted_solution_function_body = extract_function_body(corrupted_solution_with_docstring)

        test_result = safe_exec(corrupted_solution, test_code)
        assert not test_result["is_correct"], f"Generated solution for problem {task_id} unexpectedly passes all tests!"

        corrupted_solutions.append(
            {
                "task_id": task_id,
                "prompt": problem_statement,
                "canonical_solution": original_solution_body,
                "corrupted_solution": corrupted_solution_with_docstring,
                "test_code": test_code,
                "raw_llm_response": llm_response,
                "diff_count": count_diff_lines(original_solution_body, corrupted_solution_function_body),
                "edit_count": get_levenshtein_distance(original_solution_body, corrupted_solution_function_body),
            }
        )

    with open("corrupted_solutions.json", "w") as f:
        json.dump(corrupted_solutions, f, indent=2)


if __name__ == "__main__":
    main()
