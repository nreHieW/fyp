import json
import os
from datasets import load_dataset

SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement, your task is to fix the code while preserving as much of the original code as possible.
Do not change the function signature, default arguments, or docstring. Use the docstring to understand the requirements of the function.
IMPORTANT: Try to preserve the original code and the logic of the original code as much as possible."""

GENERIC_SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement, your task is to fix the code while preserving as much of the original code as possible.
Do not change the function signature, default arguments, or docstring. Use the docstring to understand the requirements of the function."""


def create_user_message(
    problem_statement,
    corrupted_solution,
):
    base_message = (
        "I am trying to implement a function with the following specifications:\n" f"{problem_statement}.\n\n" "The function I have written so far is:\n" "```python" f"{corrupted_solution} ```\n\n"
    )
    return base_message + "Wrap your response in ```python and ```"


if __name__ == "__main__":

    base_path = "LLaMA-Factory/data/"
    os.makedirs(base_path, exist_ok=True)

    for split in ["train", "test"]:
        ds = load_dataset("nreHieW/DeepCoder-Partial-Edits", split=split)
        out = []

        data_path = f"{base_path}deepcoder_partial_edits_{split}.json"

        for item in ds:
            problem_spec = item["problem"]
            corrupted_answer = item["corrupted_answer"]
            correct_answer = item["correct_answer"]

            user_message = create_user_message(problem_spec, corrupted_answer)
            out.append(
                {
                    "conversations": [{"from": "user", "content": user_message}, {"from": "assistant", "content": correct_answer}],
                    # "system": SYSTEM_PROMPT,
                    "system": GENERIC_SYSTEM_PROMPT,
                }
            )

        with open(data_path, "w") as f:
            json.dump(out, f, indent=2)

    dataset_info = {
        "deepcoder_partial_edits_train": {
            "file_name": f"deepcoder_partial_edits_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {"role_tag": "from", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        },
        "deepcoder_partial_edits_test": {
            "file_name": f"deepcoder_partial_edits_test.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {"role_tag": "from", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        },
    }

    with open(f"{base_path}dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
