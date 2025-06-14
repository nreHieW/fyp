import json
import os
import argparse

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from partial_edits.utils.extract_utils import *

load_dotenv()

SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement, your task is to fix the code while preserving as much of the original code as possible.
Do not change the function signature, default arguments, or docstring. Use the docstring to understand the requirements of the function.
IMPORTANT: Try to preserve the original code and the logic of the original code as much as possible."""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat-v3-0324:free", help="Model name to use for evaluation")
    parser.add_argument("--questions_path", type=str, required=True, help="Path to the JSONL file containing questions")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Load questions from JSONL file
    questions = []
    with open(args.questions_path, "r") as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    model_name = args.model.split("/")[-1]
    output_file = f"results_{model_name}.jsonl"

    # Load existing results to avoid reprocessing
    processed_task_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line.strip())
                        processed_task_ids.add(result["task_id"])
        except json.JSONDecodeError:
            print(f"Error loading {output_file}, starting fresh")

    try:
        with open(output_file, "a") as f:
            for problem in tqdm(questions):
                if problem["task_id"] in processed_task_ids:
                    continue

                problem_statement = problem["prompt"]
                corrupted_solution = problem["corrupted_solution"]
                user_message = f"I am trying to implement a function with the following specifications:\n{problem_statement}.\n\nThe function I have written so far is:\n{corrupted_solution}. \n\n What is wrong? Fix and complete my function but keep as much of the original code as possible."
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": user_message,
                        },
                    ],
                )
                llm_response = completion.choices[0].message.content
                llm_solution = extract_code_from_response(llm_response)

                result_entry = {
                    "task_id": problem["task_id"],
                    "solution": llm_solution,  # LLM's fixed solution for evaluation
                    "llm_response": llm_response,  # Full LLM response
                    "corrupted_solution": corrupted_solution,  # Original buggy code
                    "prompt": problem_statement,  # Problem description
                    "canonical_solution": problem["canonical_solution"],
                    "test_code": problem["test_code"],
                }

                f.write(json.dumps(result_entry) + "\n")
                f.flush()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
