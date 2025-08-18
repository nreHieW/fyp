import json
import os
import sys
import argparse
import random
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from icl_code.utils.prompt_utils import create_generation_system_prompt, create_generation_user_message
from icl_code.utils.extract_utils import extract_code_from_response
from partial_edits.utils.extract_utils import extract_docstring


def load_style_guide(style_guide_path: str) -> str:
    """Load the Astra-Prime style guide from file."""
    with open(style_guide_path, "r") as f:
        return f.read()


def generate_styled_examples(problems: List[Dict], style_guide: str, model_name: str, is_reasoning: bool = False, n_examples: int = 20, seed: int = 42) -> Tuple[List[Dict], str]:
    # Sample n_examples problems randomly
    random.seed(seed)
    if len(problems) < n_examples:
        print(f"Warning: Only {len(problems)} problems available, requested {n_examples}")
        n_examples = len(problems)

    sampled_problems = random.sample(problems, n_examples)
    print(f"Sampled {n_examples} problems from {len(problems)} available (seed: {seed})")

    model = get_model(model_name, is_reasoning)

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
            }
            examples.append(example)

        print(f"Successfully generated {len(examples)} examples")

    except Exception as e:
        print(f"Error in batch generation: {e}")
        return []

    return examples, system_prompt


def main():
    parser = argparse.ArgumentParser(description="Generate styled code examples and test questions")
    parser.add_argument("--model", required=True, help="Model name to use for generation")
    parser.add_argument("--is_reasoning", action="store_true", help="Whether to use reasoning model")
    parser.add_argument("--style_guide_path", default="data/icl_code/style_guide.txt", help="Path to style guide")
    parser.add_argument("--examples_output", default="data/icl_code/styled_examples.jsonl", help="Output path for styled examples")
    parser.add_argument("--questions_output", default="data/questions/test_questions.jsonl", help="Output path for test questions")
    parser.add_argument("--n_examples", type=int, default=20, help="Number of styled examples to generate")
    parser.add_argument("--n_questions", type=int, default=400, help="Number of test questions to prepare")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--hard", action="store_true", help="Use BigCodeBench-Hard dataset")

    args = parser.parse_args()

    # Load style guide
    if not os.path.exists(args.style_guide_path):
        print(f"Error: Style guide not found at {args.style_guide_path}")
        sys.exit(1)

    style_guide = load_style_guide(args.style_guide_path)
    print(f"Loaded style guide from {args.style_guide_path}")

    # Load BigCodeBench dataset
    dataset_name = "bigcode/bigcodebench-hard" if args.hard else "bigcode/bigcodebench"
    print(f"Loading dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name, split="v0.1.4", cache_dir="data")
        problems = list(dataset)
        print(f"Loaded {len(problems)} problems from {dataset_name}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Split problems into examples and questions (no overlap)
    random.seed(args.seed)
    shuffled_problems = random.sample(problems, len(problems))
    example_problems = shuffled_problems[: args.n_examples]
    question_problems = shuffled_problems[args.n_examples : args.n_examples + args.n_questions]

    print(f"Split problems: {len(example_problems)} for examples, {len(question_problems)} for questions")

    # Check for existing examples to resume
    examples_path = args.examples_output
    processed_task_ids = set()
    examples = []
    metadata = {}

    if os.path.exists(examples_path):
        try:
            with open(examples_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        if "_metadata" not in data:
                            examples.append(data)
                            processed_task_ids.add(data["task_id"])
                        else:
                            metadata = data
            print(f"Found existing examples for {len(processed_task_ids)} tasks")
        except Exception as e:
            print(f"Error loading existing examples: {e}")

    problems_to_process = [p for p in example_problems if p["task_id"] not in processed_task_ids]
    if len(problems_to_process) > 0:
        print(f"\n=== Generating Styled Examples ({len(problems_to_process)} remaining) ===")
        new_examples, system_prompt = generate_styled_examples(
            problems=problems_to_process, style_guide=style_guide, model_name=args.model, is_reasoning=args.is_reasoning, n_examples=len(problems_to_process), seed=args.seed
        )
        examples.extend(new_examples)

        metadata = {
            "generation_model": args.model,
            "is_reasoning": args.is_reasoning,
            "style_guide_path": args.style_guide_path,
            "dataset": dataset_name,
            "seed": args.seed,
            "total_examples": len(examples),
            "purpose": "styled_examples",
            "system_prompt": system_prompt,
        }

        os.makedirs(os.path.dirname(examples_path), exist_ok=True)

        with open(examples_path, "w") as f:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
            for example in examples:
                f.write(json.dumps(example) + "\n")

    questions_path = args.questions_output
    questions_metadata = {"dataset": dataset_name, "seed": args.seed, "total_questions": len(question_problems), "purpose": "test_questions", "style_guide_path": args.style_guide_path}

    os.makedirs(os.path.dirname(questions_path), exist_ok=True)

    with open(questions_path, "w") as f:
        f.write(json.dumps({"_metadata": questions_metadata}) + "\n")
        for question in question_problems:
            question_entry = {
                "task_id": question["task_id"],
                "problem": question["instruct_prompt"],
                "test_code": question.get("test", ""),
                "canonical_solution": question.get("canonical_solution", ""),
            }
            f.write(json.dumps(question_entry) + "\n")

    print(f"\n=== Generation completed successfully! ===")
    print(f"Generated {len(examples)} styled examples out of {args.n_examples} requested")
    print(f"Prepared {len(question_problems)} test questions out of {args.n_questions} requested")
    print(f"Styled examples: {examples_path}")
    print(f"Test questions: {questions_path}")


if __name__ == "__main__":
    main()
