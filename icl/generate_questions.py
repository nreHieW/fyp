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


def generate_questions(questions: List[Dict], seed: int, classification_function: str = "default", n: int = None, balance_first_20: bool = False) -> List[Dict]:
    """
    Generate ICL classification questions from corrupted solutions.

    Args:
        questions: List of question dictionaries
        canonical_solutions: Dictionary mapping task_id to canonical solution
        problems: Dictionary mapping task_id to problem specification
        classification_function: Name of classification function to use
        seed: Random seed for sampling
        n: Number of samples to randomly select (if None, use all)

    Returns:
        List[Dict]: List of generated ICL questions
    """
    print(f"Using classification function: {classification_function}")
    classification_func = get_classification_function(classification_function)

    # Build questions, then enforce 50-50 by sampling equal counts per class
    all_questions = []
    for question in questions:
        canonical_solution_body = question["canonical_solution"]
        function_signature_with_docstring = question["complete_prompt"]
        complete_canonical_solution = function_signature_with_docstring + "\n" + canonical_solution_body

        label = classification_func(complete_canonical_solution).value
        icl_question = {"task_id": question["task_id"], "sample": complete_canonical_solution, "ground_truth": label}
        all_questions.append(icl_question)

    # Enforce 50-50 distribution via stratified sampling
    class_a = [q for q in all_questions if q["ground_truth"] == Label.CLASS_A.value]
    class_b = [q for q in all_questions if q["ground_truth"] == Label.CLASS_B.value]

    if seed is not None:
        random.seed(seed)

    if n is not None:
        total_needed = n + 20
        if total_needed % 2 != 0:
            print(f"Requested total {total_needed} is odd; reducing by 1 to preserve 50-50 balance.")
            total_needed -= 1
        per_class = total_needed // 2
        if per_class > len(class_a) or per_class > len(class_b):
            raise ValueError(f"Not enough samples to ensure 50-50 balance: need {per_class} per class, " f"have A={len(class_a)}, B={len(class_b)}")
        icl_questions = random.sample(class_a, per_class) + random.sample(class_b, per_class)
        random.shuffle(icl_questions)
        print(f"Stratified sampled {per_class} per class -> {total_needed} total (n={n} + 20 ICL examples) (seed: {seed})")
    else:
        # Use all but shuffle
        icl_questions = class_a + class_b
        random.shuffle(icl_questions)
        print(f"Shuffled all {len(icl_questions)} questions (seed: {seed})")

    print(f"Generated {len(icl_questions)} ICL questions")

    # Print class distribution
    class_counts = {}
    for q in icl_questions:
        class_label = q["ground_truth"]
        class_counts[class_label] = class_counts.get(class_label, 0) + 1

    print("Class distribution:")
    for class_label, count in class_counts.items():
        print(f"  {class_label}: {count}")

    # Optionally balance the first 20 as A B A B ... for few-shot examples
    if balance_first_20 and len(icl_questions) >= 20:
        print("Balancing first 20 samples for few-shot examples...")
        class_a_questions = [q for q in icl_questions if q["ground_truth"] == Label.CLASS_A.value]
        class_b_questions = [q for q in icl_questions if q["ground_truth"] == Label.CLASS_B.value]
        if len(class_a_questions) >= 10 and len(class_b_questions) >= 10:
            balanced_first_20 = []
            for i in range(10):
                balanced_first_20.append(class_a_questions[i])
                balanced_first_20.append(class_b_questions[i])
            remaining_questions = [q for q in icl_questions if q not in balanced_first_20]
            icl_questions = balanced_first_20 + remaining_questions
            print("First 20 samples are now balanced: A B A B ...")
        else:
            print("Warning: Not enough samples of each class to balance first 20. Using original order.")

    return icl_questions


def main():
    parser = argparse.ArgumentParser(description="Generate ICL Classification Questions")
    parser.add_argument("--classification_function", default="default", help=f"Classification function to use. Available: {list_classification_functions()}")
    parser.add_argument("--seed", type=int, help="Random seed for sampling", default=42)
    parser.add_argument("--n", type=int, help="Number of samples to randomly select (if not specified, use all)")
    parser.add_argument("--output_path", help="Path to output file", required=True)
    parser.add_argument("--balance_first_20", action="store_true", help="Balance the first 20 samples for few-shot examples (A B ...)")

    args = parser.parse_args()

    questions = load_dataset("bigcode/bigcodebench", split="v0.1.0_hf")

    icl_questions = generate_questions(
        questions=questions,
        classification_function=args.classification_function,
        seed=args.seed,
        n=args.n,
        balance_first_20=args.balance_first_20,
    )

    # Save questions
    os.makedirs("data/questions", exist_ok=True)
    output_path = os.path.join("data/questions", args.output_path)

    metadata = {
        "classification_function": args.classification_function,
        "seed": args.seed,
        "total_questions": len(icl_questions),
        "reserved_few_shot": 20,
        "n_requested": args.n,
        "balance_first_20": args.balance_first_20,
    }

    with open(output_path, "w") as f:
        # Write metadata as first line
        f.write(json.dumps({"_metadata": metadata}) + "\n")

        # Write questions
        for question in icl_questions:
            f.write(json.dumps(question) + "\n")

    print(f"Questions saved to {output_path}")
    print(f"\nQuestion generation completed successfully!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
