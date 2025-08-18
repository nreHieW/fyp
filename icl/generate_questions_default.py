import json
import os
import sys
import argparse
import random
from typing import Dict, List
from datasets import load_dataset

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icl.utils.classification_functions import get_classification_function, list_classification_functions, Label


def load_questions(file_path: str) -> List[Dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))
    return questions


def load_canonical_solutions(questions: List[Dict]) -> Dict[str, str]:
    """Load canonical solutions from BigCodeBench dataset."""
    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4", cache_dir="data")
    canonical_solutions = {}

    for item in dataset:
        if item["task_id"] in [q["task_id"] for q in questions]:
            canonical_solution_body = item["canonical_solution"]
            function_signature_with_docstring = item["complete_prompt"]
            complete_canonical_solution = function_signature_with_docstring + "\n" + canonical_solution_body
            canonical_solutions[item["task_id"]] = complete_canonical_solution

    return canonical_solutions


def load_problems(questions: List[Dict]) -> Dict[str, Dict]:
    """Load problem specifications from BigCodeBench dataset."""
    # Determine if we need hard dataset based on task_ids
    task_ids = [q["task_id"] for q in questions]
    uses_hard = any("hard" in task_id.lower() for task_id in task_ids)

    if uses_hard:
        dataset_name = "bigcode/bigcodebench-hard"
    else:
        dataset_name = "bigcode/bigcodebench"

    dataset = load_dataset(dataset_name, split="v0.1.4", cache_dir="data")
    problems = {}

    for item in dataset:
        if item["task_id"] in [q["task_id"] for q in questions]:
            problems[item["task_id"]] = item

    return problems


def generate_questions(
    corrupted_solutions: List[Dict], canonical_solutions: Dict[str, str], seed: int, classification_function: str = "default", n: int = None, balance_first_20: bool = False
) -> List[Dict]:
    """
    Generate ICL classification questions from corrupted solutions.

    Args:
        questions: List of question dictionaries
        canonical_solutions: Dictionary mapping task_id to canonical solution
        problems: Dictionary mapping task_id to problem specification
        classification_function: Name of classification function to use
        seed: Random seed for sampling
        n: Number of samples to randomly select (if None, use all)
        balance_first_20: If True, ensure first 20 samples are balanced A B A B ...

    Returns:
        List[Dict]: List of generated ICL questions
    """
    corrupted_solutions = [q for q in corrupted_solutions if q["task_id"] in canonical_solutions]
    print(f"Processing {len(corrupted_solutions)} corrupted solutions with canonical solutions")

    # Get classification function
    print(f"Using classification function: {classification_function}")
    classification_func = get_classification_function(classification_function)

    # Build paired questions per task to guarantee 50-50 when sampling
    pairs = []
    for item in corrupted_solutions:
        task_id = item["task_id"]
        corrupted_sample = item["corrupted_solution"]
        canonical_sample = canonical_solutions[task_id]

        a_q = {"task_id": task_id + "_corrupted", "sample": corrupted_sample, "ground_truth": "CLASS_A"}
        b_q = {"task_id": task_id + "_canonical", "sample": canonical_sample, "ground_truth": "CLASS_B"}
        pairs.append([a_q, b_q])

    random.seed(seed)
    if n is not None:
        # Generate exactly n + 20 items from pairs to ensure 50-50 distribution
        total_needed = n + 20
        if total_needed % 2 != 0:
            print(f"Requested total {total_needed} is odd; reducing by 1 to preserve 50-50 balance.")
            total_needed -= 1
        pairs_needed = total_needed // 2
        if pairs_needed > len(pairs):
            raise ValueError(f"Not enough pairs available: requested {pairs_needed} pairs, only {len(pairs)} present.")
        selected_pairs = random.sample(pairs, pairs_needed)
        # Flatten selected pairs and shuffle for random order (first 20 optionally re-balanced below)
        icl_questions = [q for pair in selected_pairs for q in pair]
        random.shuffle(icl_questions)
        print(f"Randomly sampled {pairs_needed} pairs -> {total_needed} questions (n={n} + 20 ICL examples) (seed: {seed})")

    # Balance first 20 samples if requested
    if balance_first_20 and len(icl_questions) >= 20:
        print("Balancing first 20 samples for few-shot examples...")

        # Separate questions by class
        class_a_questions = [q for q in icl_questions if q["ground_truth"] == Label.CLASS_A.value]
        class_b_questions = [q for q in icl_questions if q["ground_truth"] == Label.CLASS_B.value]

        # Ensure we have enough of each class
        if len(class_a_questions) >= 10 and len(class_b_questions) >= 10:
            # Create balanced first 20: A B A B ... (10 pairs)
            balanced_first_20 = []
            for i in range(10):
                balanced_first_20.append(class_a_questions[i])
                balanced_first_20.append(class_b_questions[i])

            # Get remaining questions (excluding the ones used in first 20)
            remaining_questions = [q for q in icl_questions if q not in balanced_first_20]

            # Combine balanced first 20 with remaining questions
            icl_questions = balanced_first_20 + remaining_questions
            print("First 20 samples are now balanced: A B A B ...")
        else:
            print("Warning: Not enough samples of each class to balance first 20. Using original order.")

    print(f"Generated {len(icl_questions)} ICL questions")

    # Print class distribution
    class_counts = {}
    for q in icl_questions:
        class_label = q["ground_truth"]
        class_counts[class_label] = class_counts.get(class_label, 0) + 1

    print("Class distribution:")
    for class_label, count in class_counts.items():
        print(f"  {class_label}: {count}")

    return icl_questions


def main():
    parser = argparse.ArgumentParser(description="Generate ICL Classification Questions")
    parser.add_argument("--questions_path", help="Path to input JSONL file with corrupted solutions")
    parser.add_argument("--classification_function", default="default", help=f"Classification function to use. Available: {list_classification_functions()}")
    parser.add_argument("--seed", type=int, help="Random seed for sampling", default=42)
    parser.add_argument("--n", type=int, help="Number of samples to randomly select (if not specified, use all)")
    parser.add_argument("--balance_first_20", action="store_true", help="Balance the first 20 samples for few-shot examples (A B ...)")

    args = parser.parse_args()

    assert args.questions_path.endswith(".jsonl"), "Input must be a JSONL file"

    print(f"Loading questions from {args.questions_path}...")
    corrupted_solutions = load_questions(args.questions_path)
    print(f"Loaded {len(corrupted_solutions)} corrupted solutions")

    canonical_solutions = load_canonical_solutions(corrupted_solutions)

    icl_questions = generate_questions(
        corrupted_solutions=corrupted_solutions,
        canonical_solutions=canonical_solutions,
        classification_function=args.classification_function,
        seed=args.seed,
        n=args.n,
        balance_first_20=args.balance_first_20,
    )
    output_filename = f"icl_default_{len(icl_questions)}{'_balanced' if args.balance_first_20 else ''}.jsonl"

    # Save questions
    os.makedirs("data/questions", exist_ok=True)
    output_path = os.path.join("data/questions", output_filename)

    metadata = {
        "classification_function": args.classification_function,
        "seed": args.seed,
        "total_questions": len(icl_questions),
        "balance_first_20": args.balance_first_20,
        "reserved_few_shot": 20,
        "n_requested": args.n,
    }

    with open(output_path, "w") as f:
        # Write metadata as first line
        f.write(json.dumps({"_metadata": metadata}) + "\n")

        # Write questions
        for question in icl_questions:
            # Serialize enum as value
            q = question.copy()
            if isinstance(q["ground_truth"], Label):
                q["ground_truth"] = q["ground_truth"].value
            f.write(json.dumps(q) + "\n")

    print(f"Questions saved to {output_path}")
    print(f"\nQuestion generation completed successfully!")
    print(f"Output: {output_path}")

    # Print labels of the first 3 questions
    print(f"\nFirst 20 questions:")
    for i, question in enumerate(icl_questions[:20], 1):
        label = question["ground_truth"]
        if isinstance(label, Label):
            label = label.value
        print(f"  {i}. Task {question['task_id']}: {label}")


if __name__ == "__main__":
    main()
