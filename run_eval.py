import json
import os
import argparse

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from partial_edits.utils.code_utils import safe_exec
from partial_edits.utils.extract_utils import *

load_dotenv()

SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement, your task is to fix the code while preserving as much of the original code as possible.
Do not change the function signature, default arguments, or docstring. Use the docstring to understand the requirements of the function.
IMPORTANT: Try to preserve the original code and the logic of the original code as much as possible."""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat-v3-0324:free", help="Model name to use for evaluation")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    data = load_dataset("bigcode/bigcodebench-hard", cache_dir="data", split="v0.1.0_hf")
    model_name = args.model.split("/")[-1]
    output_file = f"results_{model_name}.json"

    out = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                out = json.load(f)
        except json.JSONDecodeError:
            print(f"Error loading {output_file}, starting fresh")

    try:
        for problem in tqdm(data):
            if any(item["task_id"] == problem["task_id"] for item in out):
                continue

            problem_statement = problem["instruct_prompt"]
            problem_statement = problem_statement.split("You should write self-contained code starting with:")[0]
            original_solution_body = problem["canonical_solution"]  # only the function body
            test_code = problem["test"]

            num_lines_canonical_solution = len(original_solution_body.split("\n"))
            num_lines_to_remove = int(num_lines_canonical_solution * 0.3)
            corrupted_solution = "\n".join(original_solution_body.split("\n")[:-num_lines_to_remove])
            corrupted_solution = problem["complete_prompt"] + "\n" + corrupted_solution

            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"I am trying to implement a function with the following specifications:\n{problem_statement}. \n\n The function I have written is:\n{corrupted_solution}. \n\n What is wrong? Fix and complete my function but keep as much of the original code as possible.",
                    },
                ],
            )
            llm_response = completion.choices[0].message.content

            llm_solution = extract_code_from_response(llm_response)
            llm_solution_body = extract_function_body(llm_solution)
            llm_solution = problem["complete_prompt"] + "\n" + llm_solution_body
            test_result = safe_exec(llm_solution, test_code)
            diff_count = count_diff_lines(original_solution_body, llm_solution_body)
            edit_count = get_levenshtein_distance(original_solution_body, llm_solution_body)

            problem["llm_solution"] = llm_solution
            problem["diff_count"] = diff_count
            problem["edit_count"] = edit_count
            problem["llm_response"] = llm_response
            problem["is_correct"] = test_result["is_correct"]
            problem["corrupted_solution"] = corrupted_solution
            out.append(problem)

            with open(output_file, "w") as f:
                json.dump(out, f, indent=2)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        raise


if __name__ == "__main__":
    main()
