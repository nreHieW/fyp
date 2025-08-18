import json
import os
import sys
import argparse
from typing import Dict, List, Tuple
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.prompt_utils import create_few_shot_examples, create_system_prompt, create_user_message
from utils.extract_utils import extract_answer_from_response, CLASS_A_DESCRIPTIVE, CLASS_B_DESCRIPTIVE
from utils.classification_functions import Label


def load_questions_with_metadata(file_path: str) -> Tuple[List[Dict], Dict]:
    """Load questions from JSONL file, separating metadata from questions."""
    questions = []
    metadata = None

    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                if "_metadata" in data:
                    metadata = data["_metadata"]
                else:
                    questions.append(data)

    return questions, metadata


def evaluate(questions_path: str, num_shots: int = 0, model_name: str = None, is_reasoning: bool = False, use_descriptive_labels: bool = False, save_results: bool = True):

    assert questions_path is not None, "No questions provided"
    assert questions_path.endswith(".jsonl"), "Questions must be a JSONL file"
    assert model_name is not None, "No model provided"

    questions, metadata = load_questions_with_metadata(questions_path)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    if metadata:
        print(f"Questions generated with: {metadata.get('classification_function', 'unknown')} classification function")

    # Reserve a fixed number of few-shot examples based on metadata; evaluate the rest
    reserved_few_shot = metadata.get("reserved_few_shot", 20) if metadata else 20
    if len(questions) < reserved_few_shot:
        print(f"Warning: Only {len(questions)} questions available; need at least {reserved_few_shot} for few-shot examples.")
    few_shot_pool = questions[:reserved_few_shot]
    all_questions = questions[reserved_few_shot:]

    few_shot_examples, _ = create_few_shot_examples(few_shot_pool, num_shots, use_descriptive_labels)

    print(f"Using {num_shots} fixed few-shot examples (from the first {reserved_few_shot}). " f"Evaluating on {len(all_questions)} remaining questions from original {len(questions)}.")

    # Generate output paths using data/icl/<questions path>/<model>/{n_shots}_results.json structure
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "icl")
    questions_path_base = os.path.basename(questions_path).replace(".jsonl", "")
    questions_dir = os.path.join(data_dir, questions_path_base)
    questions_dir += "_descriptive_labels" if use_descriptive_labels else ""
    model_dir = os.path.join(questions_dir, model_name.replace("/", "_") + f"{'_reasoning' if is_reasoning else ''}")
    result_path = os.path.join(model_dir, f"{num_shots}_shots_results.json")

    if save_results:
        os.makedirs(model_dir, exist_ok=True)
    print(f"Results will be saved to: {result_path}")
    processed_task_ids = set()
    existing_results = []
    existing_system_prompt = None
    if os.path.exists(result_path):
        try:
            with open(result_path, "r") as f:
                existing_data = json.load(f)
                if "results" in existing_data:
                    existing_results = existing_data["results"]
                    # Only consider successfully completed tasks as processed
                    processed_task_ids = set(result["task_id"] for result in existing_results if result.get("is_correct") is not None and not result.get("error"))
                    print(f"Found existing results for {len(processed_task_ids)} successfully completed tasks")
                # Extract system prompt from existing data if available
                if "system_prompt" in existing_data:
                    existing_system_prompt = existing_data["system_prompt"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading existing results from {result_path}: {e}")
            print("Starting fresh evaluation")

    # Check if all questions are already processed
    all_task_ids = set(q["task_id"] for q in all_questions)
    if all_task_ids.issubset(processed_task_ids):
        print(f"All {len(all_questions)} tasks already processed successfully.")
        print("Redoing extraction for all existing results...")
        questions_to_process = []
    else:
        # Filter out already processed questions
        questions_to_process = [q for q in all_questions if q["task_id"] not in processed_task_ids]
        print(f"Skipping {len(processed_task_ids)}/{len(all_questions)} successfully completed tasks")
        print(f"Processing {len(questions_to_process)} remaining tasks")

    # Process new questions if any
    new_results = []
    if questions_to_process:
        mode_description = f"descriptive ({CLASS_A_DESCRIPTIVE}/{CLASS_B_DESCRIPTIVE})" if use_descriptive_labels else f"generic ({Label.CLASS_A.value}/{Label.CLASS_B.value})"
        print(f"Using {mode_description}")

        system_prompt = create_system_prompt(few_shot_examples, use_descriptive_labels)

        model = get_model(model_name, is_reasoning)

        user_messages = []
        question_data = []

        for question in questions_to_process:
            task_id = question["task_id"]
            sample = question["sample"]  # The complete solution sample to classify
            ground_truth = question["ground_truth"]
            # Convert ground truth to appropriate format for descriptive labels
            # Handle both enum and string values for ground_truth
            if isinstance(ground_truth, Label):
                ground_truth_value = ground_truth.value
            else:
                ground_truth_value = str(ground_truth)

            if use_descriptive_labels:
                if ground_truth_value == Label.CLASS_A.value:
                    display_ground_truth = CLASS_A_DESCRIPTIVE
                elif ground_truth_value == Label.CLASS_B.value:
                    display_ground_truth = CLASS_B_DESCRIPTIVE
                else:
                    display_ground_truth = ground_truth_value  # Keep as is if already descriptive
            else:
                if ground_truth_value == CLASS_A_DESCRIPTIVE:
                    display_ground_truth = Label.CLASS_A.value
                elif ground_truth_value == CLASS_B_DESCRIPTIVE:
                    display_ground_truth = Label.CLASS_B.value
                else:
                    display_ground_truth = ground_truth_value  # Keep as is if already generic

            user_message = create_user_message(sample, use_descriptive_labels)
            user_messages.append(user_message)

            # print(system_prompt)
            # print("-" * 100)
            # print(user_message)
            # print("*** SOLUTION *** ")
            # print(display_ground_truth)
            # exit()

            question_data.append({"task_id": task_id, "sample": sample, "ground_truth": display_ground_truth})

        try:
            batch_responses = model.generate_responses(system_prompt, user_messages)

            for i, (question_info, response) in enumerate(zip(question_data, batch_responses)):
                if response.get("error"):
                    print(f"Error in response for {question_info['task_id']}: {response['error']}")
                    continue

                llm_response = response.get("final_answer", "")
                reasoning = response.get("reasoning", "")

                extracted_answer = extract_answer_from_response(llm_response, is_reasoning, use_descriptive_labels)
                is_correct = extracted_answer.upper() == question_info["ground_truth"]
                extraction_failed = extracted_answer == ""

                result = {
                    "task_id": question_info["task_id"],
                    "sample": question_info["sample"],
                    "ground_truth": question_info["ground_truth"],
                    "llm_response": llm_response,
                    "reasoning": reasoning,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "extraction_failed": extraction_failed,
                }
                new_results.append(result)

        except Exception as e:
            print(f"Error in batch API call: {e}")
            return {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "model": model_name,
                "is_reasoning": is_reasoning,
                "num_shots": num_shots,
                "total_samples": len(all_questions),
                "evaluated_samples": len(existing_results),
                "correct_predictions": sum(1 for result in existing_results if result.get("is_correct", False)),
                "accuracy": sum(1 for result in existing_results if result.get("is_correct", False)) / len(existing_results) if len(existing_results) > 0 else 0.0,
                "error": str(e),
                "results": existing_results,
                "use_descriptive_labels": use_descriptive_labels,
            }, {"accuracy": 0.0, "total_samples": len(all_questions), "evaluated_samples": len(existing_results), "correct_predictions": 0, "failed_extractions": 0, "error": str(e)}
    else:
        # All questions already processed - redo extraction from existing responses
        print(f"All {len(all_questions)} tasks already processed successfully. Re-extracting answers from existing responses...")

        system_prompt = existing_system_prompt
        # Re-extract answers from existing responses and update results
        updated_results = []
        for result in existing_results:
            updated_result = result.copy()
            llm_response = result.get("llm_response", "")
            reasoning = result.get("reasoning", "")

            extracted_answer = extract_answer_from_response(llm_response, is_reasoning, use_descriptive_labels)
            is_correct = extracted_answer.upper() == result["ground_truth"]
            extraction_failed = extracted_answer == ""

            updated_result["extracted_answer"] = extracted_answer
            updated_result["is_correct"] = is_correct
            updated_result["extraction_failed"] = extraction_failed

            updated_results.append(updated_result)

        # Replace existing results with updated ones
        existing_results = updated_results
        print(f"Updated {len(updated_results)} results with re-extracted answers")

    all_results = existing_results + new_results

    # Calculate metrics only for successfully processed results
    successful_results = [r for r in all_results if r.get("is_correct") is not None and not r.get("error")]
    correct_predictions = sum(1 for result in successful_results if result.get("is_correct", False))

    # Handle failed extractions for both new and existing results
    # For existing results that don't have extraction_failed field, infer from extracted_answer
    failed_extractions = 0
    for result in successful_results:
        if "extraction_failed" in result:
            # New results have explicit extraction_failed field
            if result["extraction_failed"]:
                failed_extractions += 1
        else:
            # Existing results without extraction_failed field - infer from extracted_answer
            if result.get("extracted_answer", "") == "":
                failed_extractions += 1

    accuracy = correct_predictions / len(successful_results) if len(successful_results) > 0 else 0.0

    evaluation_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model_name,
        "is_reasoning": is_reasoning,
        "use_descriptive_labels": use_descriptive_labels,
        "num_shots": num_shots,
        "total_samples": len(all_questions),
        "evaluated_samples": len(successful_results),
        "correct_predictions": correct_predictions,
        "failed_extractions": failed_extractions,
        "accuracy": accuracy,
        "system_prompt": system_prompt,
        "results": all_results,
    }

    if save_results:
        with open(result_path, "w") as f:
            json.dump(evaluation_data, f, indent=2)
        print(f"Results saved to {result_path}")

    print(f"\nEvaluation completed!")
    print(f"Accuracy (among evaluated samples): {accuracy:.3f} ({correct_predictions}/{len(successful_results)})")
    print(f"Successfully evaluated: {len(successful_results)}/{len(all_questions)}")
    print(f"Failed extractions: {failed_extractions}/{len(successful_results)}")

    if new_results:
        model.print_usage()

    return evaluation_data


def main():
    parser = argparse.ArgumentParser(description="ICL Classification Evaluation")
    parser.add_argument("--questions_path", required=True, help="Path to questions JSONL file")
    parser.add_argument("--num_shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--is_reasoning", action="store_true", help="Whether to use reasoning model")
    parser.add_argument(
        "--use_descriptive_labels", action="store_true", help=f"Use descriptive labels ({CLASS_A_DESCRIPTIVE}/{CLASS_B_DESCRIPTIVE}) instead of generic ({Label.CLASS_A.value}/{Label.CLASS_B.value})"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    try:
        evaluation_data = evaluate(
            questions_path=args.questions_path,
            num_shots=args.num_shots,
            model_name=args.model,
            is_reasoning=args.is_reasoning,
            use_descriptive_labels=args.use_descriptive_labels,
            save_results=not args.no_save,
        )
        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
