import json
import os
import sys
import argparse
import random
import re
from typing import Dict, List, Tuple
from datasets import load_dataset

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import get_model
from icl_code.utils.prompt_utils import create_generation_system_prompt, create_generation_user_message
from icl_code.utils.extract_utils import extract_code_from_response
from partial_edits.utils.extract_utils import extract_docstring


def load_style_guide(style_guide_path: str) -> str:
    with open(style_guide_path, "r") as f:
        return f.read()


def extract_constraints_from_style_guide(style_guide: str) -> List[Dict[str, str]]:
    """
    Extract individual constraints from the style guide.
    Each constraint is identified by a numbered rule (e.g., **1. Title:**).
    Returns list of dictionaries with number, title, and content.
    """
    constraints = []

    # Find all numbered sections with titles
    pattern = r"\*\*(\d+)\.\s*([^*]+?)\*\*"
    matches = re.findall(pattern, style_guide)

    # Split content by the same pattern to get sections
    sections = re.split(r"\*\*\d+\.\s*[^*]+?\*\*", style_guide)

    if len(sections) > 1:  # First section is before any rules
        rule_contents = sections[1:]

        for i, (rule_num, rule_title) in enumerate(matches):
            if i < len(rule_contents):
                content = rule_contents[i].strip()
                constraints.append({"number": rule_num, "title": rule_title.strip(), "content": content})

    return constraints


def create_modified_style_guide(original_style_guide: str, constraints: List[Dict[str, str]], exclude_constraint_num: str) -> str:
    """
    Create a modified style guide with one constraint removed and remaining constraints renumbered.

    Args:
        original_style_guide: The complete style guide text
        constraints: List of constraint dictionaries from extract_constraints_from_style_guide
        exclude_constraint_num: The constraint number to remove (e.g., "3")
    """
    if not exclude_constraint_num:
        return original_style_guide

    # Find the constraint to remove
    constraint_to_remove = None
    for constraint in constraints:
        if constraint["number"] == exclude_constraint_num:
            constraint_to_remove = constraint
            break

    if not constraint_to_remove:
        return original_style_guide

    # Remove the constraint section using the title information
    escaped_title = re.escape(constraint_to_remove["title"])
    pattern = rf"\*\*{re.escape(exclude_constraint_num)}\.\s*{escaped_title}\*\*.*?(?=\*\*\d+\.|$)"
    modified_guide = re.sub(pattern, "", original_style_guide, flags=re.DOTALL)

    # Clean up any multiple newlines
    modified_guide = re.sub(r"\n\s*\n\s*\n+", "\n\n", modified_guide)

    # Renumber the remaining constraints to flow consecutively
    excluded_num = int(exclude_constraint_num)

    # For each constraint number higher than the excluded one, reduce by 1
    for constraint in constraints:
        current_num = int(constraint["number"])
        if current_num > excluded_num:
            old_num = str(current_num)
            new_num = str(current_num - 1)

            # Replace **N.** with **(N-1).**
            pattern = rf"\*\*{re.escape(old_num)}\."
            replacement = f"**{new_num}."
            modified_guide = re.sub(pattern, replacement, modified_guide)

    return modified_guide.strip()


def generate_constraint_examples(
    problems: List[Dict], style_guide: str, model_name: str, is_reasoning: bool = False, n_examples: int = 10, seed: int = 42, constraint_variation: str = "all"
) -> Tuple[List[Dict], str]:
    """
    Generate examples with specific constraint variations.

    Args:
        problems: List of problems to solve
        style_guide: The (possibly modified) style guide to use
        model_name: Model to use for generation
        is_reasoning: Whether to use reasoning model
        n_examples: Number of examples to generate
        seed: Random seed
        constraint_variation: "all" for all constraints, or constraint number to exclude
    """
    random.seed(seed)
    if len(problems) < n_examples:
        print(f"Warning: Only {len(problems)} problems available, requested {n_examples}")
        n_examples = len(problems)

    sampled_problems = random.sample(problems, n_examples)
    print(f"Generating {n_examples} examples with constraint variation: {constraint_variation}")

    model = get_model(model_name, is_reasoning)
    # Use the provided style_guide (which may be modified to exclude a constraint)
    system_prompt = create_generation_system_prompt(style_guide)

    examples = []
    user_messages = []
    problem_data = []

    for problem in sampled_problems:
        task_id = problem["task_id"]
        docstring = extract_docstring(problem["complete_prompt"])
        problem_statement = problem["instruct_prompt"][:-3] + docstring + "\n```"
        test_code = problem.get("test", "")
        canonical_solution = problem.get("canonical_solution", "")

        user_message = create_generation_user_message(problem_statement, test_code)
        user_messages.append(user_message)

        problem_info = {"task_id": task_id, "problem": problem_statement, "test_code": test_code, "canonical_solution": canonical_solution}
        problem_data.append(problem_info)

    try:
        batch_responses = model.generate_responses(system_prompt, user_messages)

        for problem_info, response in zip(problem_data, batch_responses):
            if response.get("error"):
                print(f"Error generating example for {problem_info['task_id']}: {response['error']}")
                continue

            final_answer = response.get("final_answer", "")
            styled_solution = extract_code_from_response(final_answer)
            reasoning = response.get("reasoning", "")

            if not styled_solution.strip():
                print(f"Empty solution generated for {problem_info['task_id']}")
                continue

            example = {
                "task_id": problem_info["task_id"],
                "problem": problem_info["problem"],
                "styled_solution": styled_solution,
                "llm_response": final_answer,
                "reasoning": reasoning,
                "test_code": problem_info["test_code"],
                "canonical_solution": problem_info["canonical_solution"],
                "generation_model": model_name,
                "is_reasoning": is_reasoning,
                "constraint_variation": constraint_variation,
                "excluded_constraint": constraint_variation if constraint_variation != "all" else None,
            }
            examples.append(example)

    except Exception as e:
        print(f"Error in batch generation: {e}")
        return [], system_prompt
    return examples, system_prompt


def main():
    parser = argparse.ArgumentParser(description="Generate constraint variation examples for judge evaluation")
    parser.add_argument("--model", required=True, help="Model name to use for generation")
    parser.add_argument("--is_reasoning", action="store_true", help="Whether to use reasoning model")
    parser.add_argument("--style_guide_path", default="data/icl_code/style_guide.txt", help="Path to style guide")
    parser.add_argument("--output_dir", default="data/icl_code/eval_judge", help="Output directory for examples")
    parser.add_argument("--n_examples", type=int, default=10, help="Number of examples per constraint variation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()

    # Load style guide
    if not os.path.exists(args.style_guide_path):
        print(f"Error: Style guide not found at {args.style_guide_path}")
        sys.exit(1)

    style_guide = load_style_guide(args.style_guide_path)
    print(f"Loaded style guide from {args.style_guide_path}")

    # Extract constraints
    constraints = extract_constraints_from_style_guide(style_guide)
    print(f"Extracted {len(constraints)} constraints from style guide")
    for constraint in constraints:
        print(f"  - Constraint {constraint['number']}: {constraint['title']}")

    try:
        dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4", cache_dir="data")
        problems = list(dataset)
        print(f"Loaded {len(problems)} problems from bigcodebench")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    all_examples = []
    examples_all, _ = generate_constraint_examples(
        problems=problems, style_guide=style_guide, model_name=args.model, is_reasoning=args.is_reasoning, n_examples=args.n_examples, seed=args.seed, constraint_variation="all"
    )
    all_examples.extend(examples_all)

    for constraint in constraints:
        constraint_num = constraint["number"]
        constraint_title = constraint["title"]

        print(f"\n=== Generating examples WITHOUT constraint {constraint_num}: {constraint_title} ===")

        modified_style_guide = create_modified_style_guide(style_guide, constraints, constraint_num)

        examples_modified, _ = generate_constraint_examples(
            problems=problems,
            style_guide=modified_style_guide,
            model_name=args.model,
            is_reasoning=args.is_reasoning,
            n_examples=args.n_examples,
            seed=args.seed + int(constraint_num),  # Different seed for each variation
            constraint_variation=constraint_num,
        )
        all_examples.extend(examples_modified)

    output_path = os.path.join(args.output_dir, f"judge_evaluation_examples_{args.model.replace('/', '_')}.jsonl")

    metadata = {
        "generation_model": args.model + ("_reasoning" if args.is_reasoning else ""),
        "seed": args.seed,
        "n_examples_per_variation": args.n_examples,
        "total_examples": len(all_examples),
        "constraints": constraints,
    }

    with open(output_path, "w") as f:
        f.write(json.dumps({"_metadata": metadata}) + "\n")
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Generated {len(all_examples)} total examples to {output_path}")


if __name__ == "__main__":
    main()
