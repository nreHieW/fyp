import json
import os
import sys
import argparse
from typing import Dict, List, Tuple
from collections import Counter
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import get_model
from icl_code.eval_judge.generate_examples import load_style_guide, extract_constraints_from_style_guide
from icl_code.utils.style_compliance import get_style_compliance, create_style_judge_system_prompt, create_style_judge_user_prompt

JUDGE_MODELS = [
    {"name": "google/gemini-2.5-pro_high", "is_reasoning": True},
    {"name": "deepseek/deepseek-r1-0528", "is_reasoning": True},
    {"name": "qwen/qwen3-coder", "is_reasoning": False},
    {"name": "qwen/qwen3-235b-a22b-thinking-2507", "is_reasoning": True},
    {"name": "z-ai/glm-4.5_high", "is_reasoning": True},
]


def load_examples_with_judges(examples_path: str) -> Tuple[List[Dict], Dict]:
    examples, metadata = [], {}
    with open(examples_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                if "_metadata" in data:
                    metadata = data["_metadata"]
                else:
                    if "judges" not in data:
                        data["judges"] = {}
                    examples.append(data)
    return examples, metadata


def save_examples_with_judges(examples_path: str, examples: List[Dict], metadata: Dict):
    with open(examples_path, "w") as f:
        f.write(json.dumps({"_metadata": metadata}) + "\n")
        for example in examples:
            f.write(json.dumps(example) + "\n")


def evaluate_examples_with_judge(examples: List[Dict], judge_model_name: str, style_guide: str, constraint_names: List[str], judge_is_reasoning: bool = False) -> List[Dict]:
    examples_to_evaluate = [example for example in examples if judge_model_name not in example.get("judges", {}) or example["judges"][judge_model_name].get("evaluation_error", False)]
    examples_to_reextract = [example for example in examples if judge_model_name in example.get("judges", {}) and not example["judges"][judge_model_name].get("evaluation_error", False)]

    # Re-extract compliance from existing LLM responses
    if examples_to_reextract:
        for example in examples_to_reextract:
            existing_response = example["judges"][judge_model_name]["llm_response"]
            compliance = get_style_compliance(existing_response, constraint_names)["compliance"]
            example["judges"][judge_model_name]["compliance"] = compliance

    if not examples_to_evaluate:
        print(f"  All {len(examples)} examples already evaluated for {judge_model_name}")
        return examples

    already_evaluated = len(examples) - len(examples_to_evaluate)
    if already_evaluated > 0:
        print(f"  Resuming: {already_evaluated} already evaluated, {len(examples_to_evaluate)} remaining for {judge_model_name}")

    judge_model = get_model(judge_model_name, judge_is_reasoning)
    system_prompt = create_style_judge_system_prompt(style_guide, constraint_names)
    user_prompts = [create_style_judge_user_prompt(ex["styled_solution"], len(constraint_names)) for ex in examples_to_evaluate]

    try:
        batch_responses = judge_model.generate_responses(system_prompt, user_prompts)

        for example, response in zip(examples_to_evaluate, batch_responses):
            if "judges" not in example:
                example["judges"] = {}

            if response.get("error"):
                print(f"  Error: {response['error']}")
            else:
                compliance = get_style_compliance(response.get("final_answer", ""), constraint_names)["compliance"]
                example["judges"][judge_model_name] = {"compliance": compliance, "llm_response": response.get("final_answer", ""), "evaluation_error": False}

    except Exception as e:
        for example in examples_to_evaluate:
            if "judges" not in example:
                example["judges"] = {}
            example["judges"][judge_model_name] = {"compliance": {name: False for name in constraint_names}, "llm_response": "", "evaluation_error": True, "error_message": str(e)}

    return examples


def get_ground_truth_compliance(constraint_variation: str, constraint_names: List[str]) -> Dict[str, bool]:
    if constraint_variation == "all":
        return {name: True for name in constraint_names}
    else:
        excluded_constraint = f"rule_{constraint_variation}"
        return {name: name != excluded_constraint for name in constraint_names}


def calculate_majority_vote(examples: List[Dict], constraint_names: List[str]) -> List[Dict]:
    for example in examples:
        judges = example.get("judges", {})
        valid_judges = {name: result for name, result in judges.items() if not result.get("evaluation_error", False)}

        if not valid_judges:
            example["majority_vote"] = {name: False for name in constraint_names}
            continue

        majority_vote = {}
        for constraint in constraint_names:
            votes = [judge["compliance"].get(constraint, False) for judge in valid_judges.values()]
            majority_vote[constraint] = Counter(votes).most_common(1)[0][0] if votes else False

        example["majority_vote"] = majority_vote
        example["ground_truth"] = get_ground_truth_compliance(example["constraint_variation"], constraint_names)

    return examples


def calculate_accuracy(examples: List[Dict], constraint_names: List[str]) -> Dict[str, float]:
    all_judges = set()
    for example in examples:
        if "judges" in example:
            all_judges.update(example["judges"].keys())

    accuracies = {}
    for judge_name in all_judges:
        correct = total = 0

        num_classified_correct = 0

        for example in examples:
            judge_result = example.get("judges", {}).get(judge_name)
            if not judge_result or judge_result.get("evaluation_error", False):
                continue

            ground_truth = example.get("ground_truth", {})
            compliance = judge_result.get("compliance", {})

            for constraint in constraint_names:
                if constraint in compliance and constraint in ground_truth:
                    if compliance[constraint] == ground_truth[constraint]:
                        correct += 1
                    total += 1
            if compliance == ground_truth:
                num_classified_correct += 1

        accuracies[judge_name] = {
            "constraint_accuracy": correct / total if total > 0 else 0.0,
            "classification_accuracy": num_classified_correct / len(examples) if len(examples) > 0 else 0.0,
        }

    return accuracies


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM judges on constraint variation examples")
    parser.add_argument("--examples_path", required=True, help="Path to JSONL file containing generated examples")
    parser.add_argument("--style_guide_path", default="data/icl_code/style_guide.txt", help="Path to style guide file")
    args = parser.parse_args()

    base_path = os.path.splitext(args.examples_path)[0]
    output_path = f"{base_path}_evaluated{os.path.splitext(args.examples_path)[1]}"

    # Load existing results if output file exists, otherwise load from input
    if os.path.exists(output_path):
        print(f"Loading existing results from {output_path}")
        examples, metadata = load_examples_with_judges(output_path)
    else:
        print(f"Loading fresh examples from {args.examples_path}")
        examples, metadata = load_examples_with_judges(args.examples_path)

    style_guide = load_style_guide(args.style_guide_path)
    constraints = extract_constraints_from_style_guide(style_guide)
    constraint_names = [f"rule_{constraint['number']}" for constraint in constraints]

    for judge_config in JUDGE_MODELS:
        print(f"Evaluating with {judge_config['name']}")
        examples = evaluate_examples_with_judge(examples, judge_config["name"], style_guide, constraint_names, judge_config.get("is_reasoning", False))
        save_examples_with_judges(output_path, examples, metadata)

    examples = calculate_majority_vote(examples, constraint_names)
    save_examples_with_judges(output_path, examples, metadata)

    accuracies = calculate_accuracy(examples, constraint_names)
    for judge_name, accuracy_dict in accuracies.items():
        print(f"{judge_name}: {accuracy_dict['constraint_accuracy']:.1%}, {accuracy_dict['classification_accuracy']:.1%}")


if __name__ == "__main__":
    main()
