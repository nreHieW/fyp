# /// script
# dependencies = [
#   "transformers",
#   "peft",
# ]
# ///

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import torch
import os

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _get_target_module_keys(model):
    return [key for key in model.state_dict().keys() if any(target_module in key for target_module in TARGET_MODULES)]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--peft_model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _sanity_check(base_model, merged_model):
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()

    for key in _get_target_module_keys(base_model):
        if key not in merged_state:
            raise ValueError(f"Key {key} not found in merged model")

    for key in _get_target_module_keys(merged_model):
        if key not in base_state:
            raise ValueError(f"Key {key} not found in base model")

    for key in _get_target_module_keys(base_model):
        base_tensor = base_state[key]
        merged_tensor = merged_state[key]
        assert not torch.allclose(base_tensor, merged_tensor), f"Key {key} is close"


@torch.no_grad()
def main():
    args = get_args()
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model_copy = AutoModelForCausalLM.from_pretrained(args.base_model)
    peft_model = PeftModel.from_pretrained(base_model, args.peft_model_id)
    merged_model = peft_model.merge_and_unload()
    _sanity_check(base_model_copy, merged_model)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
