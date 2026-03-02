import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from tqdm import tqdm

from partial_edits_utils.similarity_utils import get_levenshtein_distance
from deepcoder_utils.local_verify import verify_deepcoder_local
from deepcoder_utils.legacy.deepcoder_genesys import extract_code_from_model

GENERIC_SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement, your task is to fix the code while preserving as much of the original code as possible.
Do not change the function signature, default arguments, or docstring. Use the docstring to understand the requirements of the function."""


def process_test(test: str) -> dict:
    tests = json.loads(test)
    if isinstance(tests, dict):
        inputs = tests["inputs"]
        outputs = tests["outputs"]
        tests = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]

    else:
        for test in tests:
            if "inputs" in test:
                test["input"] = test.pop("inputs")

            if "outputs" in test:
                test["output"] = test.pop("outputs")

    return json.dumps(tests)


if __name__ == "__main__":

    fpaths = [
        "/scratch/e0968774/deepcoder_partial_edits_non_ood_train.json",
        "/scratch/e0968774/deepcoder_partial_edits_ood_train.json",
        "/scratch/e0968774/deepcoder_partial_edits_both_train.json",
    ]
    seed = 42

    for fpath in fpaths:
        with open(fpath, "r") as f:
            data = json.load(f)
            out = []
            for item in tqdm(data, desc="Processing items"):
                problem_spec = item["problem_spec"]
                corrupted_answer = item["corrupted_answer"]
                samples = item["samples"]
                canonical_solution = item["correct_answer"]
                baseline_distance = get_levenshtein_distance(canonical_solution, corrupted_answer)

                for completion in samples:
                    extracted_completion = extract_code_from_model(completion)
                    verification_info = {
                        "tests": process_test(item["tests"]),
                        "dataset_type": "primeintellect",
                    }

                    execution_reward = verify_deepcoder_local(
                        completion=extracted_completion,
                        verification_info=verification_info,
                        timeout_per_test=20,
                        max_tests=5,
                    )

                    if execution_reward != 1:
                        continue

                    completion_distance = get_levenshtein_distance(canonical_solution, extracted_completion)
                    levenshtein = completion_distance - baseline_distance

                    out.append(
                        {
                            "problem_spec": problem_spec,
                            "corrupted_answer": corrupted_answer,
                            "completion": completion,
                            "canonical_solution": canonical_solution,
                            "extracted_completion": extracted_completion,
                            "levenshtein": levenshtein,
                        }
                    )

        dataset = Dataset.from_list(out).shuffle(seed=seed)

        test_dataset = dataset.select(range(100))
        train_dataset = dataset.select(range(100, len(dataset)))
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        ds_stem = Path(fpath).stem  # e.g. deepcoder_partial_edits_non_ood_train
        subset = ds_stem.replace("deepcoder_partial_edits_", "").replace("_train", "")
        hub_dataset_id = f"nreHieW/DeepCoder-Partial-Edits-Synth{'-' + subset if subset != 'non_ood' else ''}"

        dataset_dict.push_to_hub(hub_dataset_id)
        print(
            "Pushed dataset to {hub_dataset_id}: {train_count} train / {test_count} test samples".format(
                hub_dataset_id=hub_dataset_id,
                train_count=len(train_dataset),
                test_count=len(test_dataset),
            )
        )
