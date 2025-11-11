import json
import os
from datasets import load_dataset
from collections import defaultdict

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
        "I am trying to implement a function with the following specifications:\n" f"{problem_statement}.\n\n" "The function I have written so far is:\n" "```python\n" f"{corrupted_solution} ```\n\n"
    )
    return base_message + "Wrap your response in ```python and ```"


if __name__ == "__main__":

    base_path = "LLaMA-Factory/data/"
    os.makedirs(base_path, exist_ok=True)

    # All dataset variants pushed by create_dataset.py
    dataset_variants = {
        "non_ood": "nreHieW/DeepCoder-Partial-Edits",
        "ood": "nreHieW/DeepCoder-Partial-Edits-ood",
        "both": "nreHieW/DeepCoder-Partial-Edits-both",
    }

    dataset_info = {}

    for variant, hf_id in dataset_variants.items():
        for split in ["train", "test"]:
            ds = load_dataset(hf_id, split=split)
            out = []

            dataset_key = f"deepcoder_partial_edits_{variant}_{split}"
            file_name = f"deepcoder_partial_edits_{variant}_{split}.json"

            data_path = f"{base_path}{file_name}"

            for item in ds:
                problem_spec = item["problem_spec"]
                corrupted_answer = item["corrupted_answer"]
                correct_answer = item["correct_answer"]

                user_message = create_user_message(problem_spec, corrupted_answer)
                out.append(
                    {
                        "conversations": [
                            {"from": "user", "content": user_message},
                            {"from": "assistant", "content": correct_answer},
                        ],
                        # "system": SYSTEM_PROMPT,
                        "system": GENERIC_SYSTEM_PROMPT,
                    }
                )

            with open(data_path, "w") as f:
                json.dump(out, f, indent=2)

            dataset_info[dataset_key] = {
                "file_name": file_name,
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "system": "system"},
                "tags": {
                    "role_tag": "from",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }

    synthetic_dataset_variants = {
        "non_ood": "nreHieW/DeepCoder-Partial-Edits-Synth",
        "ood": "nreHieW/DeepCoder-Partial-Edits-Synth-ood",
        "both": "nreHieW/DeepCoder-Partial-Edits-Synth-both",
    }

    for variant, hf_id in synthetic_dataset_variants.items():
        ds = load_dataset(hf_id)
        d = defaultdict(list)
        for item in ds["train"]:
            d[item["problem_spec"]].append(item)
        to_use = []
        for item in d:
            vals = d[item]
            vals = sorted(vals, key=lambda x: x["levenshtein"])
            to_use.extend(vals[:3])

        for split, tmp in [("train", to_use), ("test", ds["test"])]:
            out = []
            for item in tmp:
                problem_spec = item["problem_spec"]
                corrupted_answer = item["corrupted_answer"]
                completion = item["completion"]
                user_message = create_user_message(problem_spec, corrupted_answer)
                out.append(
                    {
                        "conversations": [
                            {"from": "user", "content": user_message},
                            {"from": "assistant", "content": completion},
                        ],
                        "system": GENERIC_SYSTEM_PROMPT,
                    }
                )
            with open(f"{base_path}synthetic_deepcoder_partial_edits_{variant}_{split}.json", "w") as f:
                json.dump(out, f, indent=2)

            dataset_info[f"synthetic_deepcoder_partial_edits_{variant}_{split}"] = {
                "file_name": f"synthetic_deepcoder_partial_edits_{variant}_{split}.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "system": "system"},
                "tags": {
                    "role_tag": "from",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }

    with open(f"{base_path}dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
