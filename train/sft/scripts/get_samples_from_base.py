# /// script
# requires-python = "==3.10"
# dependencies = [
#   "datasets",
#   "vllm",
#   "transformers==4.51.0",
# ]
# ///

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import os


# infer with explicit, train with generic (context distillation)
SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement, your task is to fix the code while preserving as much of the original code as possible.
Do not change the function signature, default arguments, or docstring. Use the docstring to understand the requirements of the function.
IMPORTANT: Try to preserve the original code and the logic of the original code as much as possible."""


def create_user_message(
    problem_statement,
    corrupted_solution,
):
    base_message = (
        "I am trying to implement a function with the following specifications:\n" f"{problem_statement}.\n\n" "The function I have written so far is:\n" "```python\n" f"{corrupted_solution} ```\n\n"
    )
    return base_message + "Wrap your response in ```python and ```"


if __name__ == "__main__":
    ds_to_infer = {
        "both": "nreHieW/DeepCoder-Partial-Edits-both",
        "ood": "nreHieW/DeepCoder-Partial-Edits-ood",
        "non_ood": "nreHieW/DeepCoder-Partial-Edits",
    }
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(n=8)

    for ds_name, ds_id in ds_to_infer.items():
        ds = load_dataset(ds_id, split="train")
        prompts = []
        for item in ds:
            problem_spec = item["problem_spec"]
            corrupted_answer = item["corrupted_answer"]
            correct_answer = item["correct_answer"]
            user_message = create_user_message(problem_spec, corrupted_answer)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            n_tokens = len(tokenizer.encode(prompt))
            if n_tokens > 8192:
                continue
            else:
                prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)
        out = []
        for i, item in enumerate(ds):
            curr_batch = outputs[i * 8 : (i + 1) * 8]
            output_texts = [o.outputs[0].text for o in curr_batch]
            out.append(
                {
                    **item,
                    "samples": output_texts,
                }
            )

        with open(f"/scratch/e0968774/deepcoder_partial_edits_{ds_name}_train.json", "w") as f:
            json.dump(out, f, indent=2)
