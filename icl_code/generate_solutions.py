import json
import os
import sys
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from icl_code.utils.prompt_utils import create_few_shot_examples, create_system_prompt, create_user_message
from icl_code.utils.extract_utils import extract_code_from_response


def load_examples_with_metadata(file_path: str) -> Tuple[List[Dict], Dict]:
    """Load examples from JSONL file, separating metadata from examples."""
    examples = []
    metadata = None

    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                if "_metadata" in data:
                    metadata = data["_metadata"]
                else:
                    examples.append(data)

    return examples, metadata


def load_test_problems(questions_path: str = None, hard: bool = False) -> List[Dict]:
    """Load test problems from file or BigCodeBench dataset."""
    if questions_path and os.path.exists(questions_path):
        # Load from prepared questions file
        questions, _ = load_examples_with_metadata(questions_path)
        print(f"Loaded {len(questions)} test problems from {questions_path}")
        return questions
    else:
        # Fallback to loading from BigCodeBench dataset
        from datasets import load_dataset

        dataset_name = "bigcode/bigcodebench-hard" if hard else "bigcode/bigcodebench"
        dataset = load_dataset(dataset_name, split="v0.1.4", cache_dir="data")
        problems = []
        for item in dataset:
            problem_entry = {"task_id": item["task_id"], "problem": item["instruct_prompt"], "test_code": item.get("test", ""), "canonical_solution": item.get("canonical_solution", "")}
            problems.append(problem_entry)
        print(f"Loaded {len(problems)} test problems from {dataset_name}")
        return problems


def generate_solutions(
    examples_path: str, num_shots: int, model_name: str, is_reasoning: bool = False, questions_path: str = None, n: int = None, hard: bool = False, save_results: bool = True, seed: int = 42
) -> str:
    """
    Generate solutions using few-shot examples.

    Args:
        examples_path: Path to JSONL file containing styled examples
        num_shots: Number of few-shot examples to use (3, 5, 8, 12, 20)
        model_name: Name of the model to use
        is_reasoning: Whether to use reasoning model
        questions_path: Path to JSONL file containing test questions (optional)
        n: Number of test problems to solve (if None, use all)
        hard: Whether to use BigCodeBench-Hard dataset (fallback if no questions_path)
        save_results: Whether to save results to file
        seed: Random seed for sampling

    Returns:
        Path to output file
    """
    assert examples_path is not None, "No examples path provided"
    assert examples_path.endswith(".jsonl"), "Examples must be a JSONL file"
    assert model_name is not None, "No model provided"
    assert num_shots in [3, 5, 8, 12, 20], f"Invalid num_shots: {num_shots}. Must be one of [3, 5, 8, 12, 20]"

    # Load examples
    examples, metadata = load_examples_with_metadata(examples_path)
    print(f"Loaded {len(examples)} examples from {examples_path}")

    if metadata:
        print(f"Examples generated with: {metadata.get('generation_model', 'unknown')} model")

    # Load test problems
    test_problems = load_test_problems(questions_path=questions_path, hard=hard)

    # Sample n problems if specified
    if n is not None:
        import random

        random.seed(seed)
        if n < len(test_problems):
            test_problems = random.sample(test_problems, n)
            print(f"Sampled {n} problems from original {len(test_problems)} available (seed: {seed})")

    print(f"Using {len(test_problems)} test problems")

    # Create few-shot examples and get remaining test problems
    few_shot_examples_str, remaining_examples = create_few_shot_examples(examples, num_shots)

    # Use test problems instead of remaining examples
    questions_to_solve = test_problems

    print(f"Using {num_shots} few-shot examples from styled dataset")
    print(f"Solving {len(questions_to_solve)} test problems")

    # Generate output path
    data_dir = os.path.join("data", "icl_code")
    examples_basename = os.path.basename(examples_path).replace(".jsonl", "")
    shots_dir = os.path.join(data_dir, f"{num_shots}_shots")

    model_suffix = f"{'reasoning' if is_reasoning else 'non_reasoning'}"
    result_filename = f"results_{model_name.replace('/', '_')}_{model_suffix}_{len(questions_to_solve)}.jsonl"
    result_path = os.path.join(shots_dir, result_filename)

    if save_results:
        os.makedirs(shots_dir, exist_ok=True)

    print(f"Results will be saved to: {result_path}")

    # Check for existing results to resume
    processed_task_ids = set()
    existing_results = {}
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    processed_task_ids.add(result["task_id"])
                    existing_results[result["task_id"]] = result
        print(f"Found existing results for {len(processed_task_ids)} tasks")

    # Check if all questions are already processed
    all_task_ids = set(q["task_id"] for q in questions_to_solve)
    if all_task_ids.issubset(processed_task_ids):
        print(f"All {len(questions_to_solve)} tasks already processed. Re-extracting code from existing responses...")

        # Re-extract code from existing responses and update results
        updated_results = []
        for question in questions_to_solve:
            task_id = question["task_id"]
            if task_id in existing_results:
                result = existing_results[task_id].copy()
                # Re-extract code from the full LLM response
                if "llm_response" in result:
                    result["solution"] = extract_code_from_response(result["llm_response"])
                    print(f"Re-extracted code for {task_id}")
                updated_results.append(result)

        # Save updated results
        if save_results:
            with open(result_path, "w") as f:
                for result in updated_results:
                    f.write(json.dumps(result) + "\n")
            print(f"Updated results saved to {result_path}")

        print("Re-extraction completed!")
        return result_path

    # Filter out already processed questions
    questions_to_process = [q for q in questions_to_solve if q["task_id"] not in processed_task_ids]

    if not questions_to_process:
        print("All questions already processed!")
        return result_path

    print(f"Processing {len(questions_to_process)} remaining questions")

    # Set up prompts
    # Create system prompt with few-shot examples
    system_prompt = create_system_prompt(few_shot_examples_str)

    # Initialize model
    model = get_model(model_name, is_reasoning)

    # Prepare batch requests
    user_messages = []
    question_data = []

    print("Preparing generation requests...")
    for question in tqdm(questions_to_process, desc="Preparing"):
        task_id = question["task_id"]
        problem_statement = question["problem"]

        user_message = create_user_message(problem_statement)
        user_messages.append(user_message)

        question_info = {"task_id": task_id, "problem": problem_statement, "test_code": question.get("test", ""), "canonical_solution": question.get("canonical_solution", "")}
        question_data.append(question_info)

    # Generate solutions
    print(f"Generating solutions for {len(user_messages)} problems...")
    try:
        batch_responses = model.generate_responses(system_prompt, user_messages)

        # Save results
        with open(result_path, "a") as f:
            for i, (question_info, response) in enumerate(zip(question_data, batch_responses)):
                if response.get("error"):
                    print(f"Error in response for {question_info['task_id']}: {response['error']}")
                    continue

                final_answer = response.get("final_answer", "")
                generated_solution = extract_code_from_response(final_answer)
                reasoning = response.get("reasoning", "")

                result_entry = {
                    "task_id": question_info["task_id"],
                    "problem": question_info["problem"],
                    "solution": generated_solution,
                    "llm_response": final_answer,
                    "reasoning": reasoning,
                    "canonical_solution": question_info["canonical_solution"],
                    "test_code": question_info["test_code"],
                    "num_shots": num_shots,
                    "examples_path": examples_path,
                    "generation_model": model_name,
                    "is_reasoning": is_reasoning,
                }

                # Add token usage if available
                token_info = response.get("token_usage", {})
                if token_info:
                    result_entry["token_usage"] = token_info

                f.write(json.dumps(result_entry) + "\n")
                f.flush()

    except Exception as e:
        print(f"Error in batch generation: {e}")
        raise

    print(f"Results saved to {result_path}")
    model.print_usage()

    return result_path


def main():
    parser = argparse.ArgumentParser(description="Generate solutions using few-shot styled examples")
    parser.add_argument("--model", required=True, help="Model name to use for generation")
    parser.add_argument("--is_reasoning", action="store_true", help="Whether to use reasoning model")
    parser.add_argument("--examples_path", required=True, help="Path to JSONL file containing styled examples")
    parser.add_argument("--questions_path", help="Path to JSONL file containing test questions (optional)")
    parser.add_argument("--num_shots", type=int, required=True, choices=[3, 5, 8, 12, 20], help="Number of few-shot examples to use")
    parser.add_argument("--n", type=int, help="Number of test problems to solve (if not specified, use all)")
    parser.add_argument("--hard", action="store_true", help="Use BigCodeBench-Hard dataset for test problems (fallback)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    if not os.path.exists(args.examples_path):
        print(f"Error: Examples file not found at {args.examples_path}")
        sys.exit(1)

    try:
        result_path = generate_solutions(
            examples_path=args.examples_path,
            num_shots=args.num_shots,
            model_name=args.model,
            is_reasoning=args.is_reasoning,
            questions_path=args.questions_path,
            n=args.n,
            hard=args.hard,
            save_results=not args.no_save,
            seed=args.seed,
        )

        print(f"\nSolution generation completed successfully!")
        print(f"Output: {result_path}")

    except Exception as e:
        print(f"Error during solution generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
