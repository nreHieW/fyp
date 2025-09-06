import json
import random
from typing import Any

from datasets import load_dataset, concatenate_datasets

from deepcoder_utils.local_verify import verify_deepcoder_local
from partial_edits_utils.code_corruptor import CodeCorruptor
from partial_edits_utils.prompt_utils import create_user_message


def _process_item(item: dict[str, Any], dataset_type: str) -> dict[str, Any]:
    """Transform a single DeepCoder sample into our partial-edit dataset schema.

    - Corrupt the canonical solution
    - Ensure the corrupted solution fails provided tests
    - Build the final record fields
    """
    original_problem_description = item["problem"]
    problem_spec = (
        original_problem_description.replace("Now solve the problem and return the code.", "").replace("Solve the following coding problem using the programming language python:", "").strip()
    )

    canonical_solution = item["solutions"][0]
    cleaned_canonical_solution = canonical_solution.replace("```python", "").replace("```", "").strip()

    corrupted_solution, _ = CodeCorruptor().corrupt_function(cleaned_canonical_solution, max_mutations=20, use_ood=True)

    # Ensure corrupted solution fails the provided tests; otherwise mark as None to drop later
    try:
        tests_obj = json.loads(item["tests"]) if isinstance(item["tests"], str) else item["tests"]
        if isinstance(tests_obj, list):
            total_tests = len(tests_obj)
        elif isinstance(tests_obj, dict):
            if "inputs" in tests_obj:
                total_tests = len(tests_obj["inputs"])
            elif "input_output" in tests_obj:
                io = json.loads(tests_obj["input_output"])
                total_tests = len(io.get("inputs", []))
            else:
                total_tests = 5
        else:
            total_tests = 5

        verification_info = {"dataset_type": dataset_type, "ground_truth": item["tests"]}
        result = verify_deepcoder_local(
            completion=corrupted_solution if corrupted_solution is not None else "",
            verification_info=verification_info,
            timeout_per_test=30,
            max_tests=total_tests,
        )
        if result == 1:
            corrupted_solution = None
    except Exception:
        # On any verification error, mark as invalid to be filtered out
        corrupted_solution = None

    return {
        # "user_message": create_user_message(problem_spec, corrupted_solution),
        "problem_spec": problem_spec,
        "correct_answer": canonical_solution,
        "corrupted_answer": corrupted_solution,
        "tests": item["tests"],
    }


def main():
    seed = 42
    random.seed(seed)

    ds_list = []
    for dataset_type in ["primeintellect", "taco"]:
        ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", dataset_type, split="train")
        ds = ds.map(_process_item, num_proc=8, fn_kwargs={"dataset_type": dataset_type}).select_columns(["problem_spec", "correct_answer", "corrupted_answer", "tests"])
        ds = ds.filter(lambda x: x["corrupted_answer"] is not None)
        ds_list.append(ds)

    ds_final = concatenate_datasets(ds_list)
    ds_dict = ds_final.train_test_split(test_size=0.1, seed=seed)
    print(ds_dict)
    ds_dict.push_to_hub("nreHieW/DeepCoder-Partial-Edits")


if __name__ == "__main__":
    main()
