import json
import os
import argparse
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.extract_utils import *
from utils.prompts_utils import generate_shots, create_user_message, get_system_prompt_with_shots
from models import get_model
from evaluate import SimilarityCalculator, _process_similarity_metrics, _calculate_corpus_metrics, _compute_difference_with_canonical

load_dotenv(override=True)


def _generate_output_path(args, questions):
    model_name = args.model.split("/")[-1]
    base_dir = "data/code_edits"

    suffixes = ["_explicit" if args.is_explicit else "", "_with_tests" if args.include_test_cases else ""]
    suffix = "".join(suffixes)

    reasoning_type = "reasoning" if args.is_reasoning else "non_reasoning"
    filename = f"results_{model_name}_{reasoning_type}{suffix}_{len(questions)}.jsonl"

    if args.num_shots > 0:
        shot_dir = f"{args.num_shots}_shot{suffix}"
        return os.path.join(base_dir, shot_dir, filename)
    else:
        if args.generic:
            return os.path.join(base_dir, "zero_shot_generic", filename)
        else:
            return os.path.join(base_dir, filename)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to use for evaluation")
    parser.add_argument("--is_reasoning", action="store_true", help="Whether to use reasoning model")
    parser.add_argument("--questions_path", type=str, required=True, help="Path to the JSONL file containing questions")
    parser.add_argument("--num_shots", type=int, default=0, help="Number of shots to use for few-shot learning")
    parser.add_argument("--is_explicit", action="store_true", help="Whether to end the user message with the instruction")
    parser.add_argument("--include_test_cases", action="store_true", help="Whether to include test cases in the user message")
    parser.add_argument("--generic", action="store_true", help="Use generic mode without minimal editing instructions (in system prompt) (only allowed with num_shots=0)")
    parser.add_argument("--store_token_info", action="store_true", help="Whether to store token usage information on a per-sample basis")
    args = parser.parse_args()

    if args.generic and args.num_shots > 0:
        parser.error("--generic flag can only be used with num_shots=0")  # because few shot with generic is equivalent to few shot + explicit

    if args.generic and args.is_explicit:
        parser.error("--generic flag cannot be used with --is_explicit")

    return args


def _prepare_batch_data(questions, processed_task_ids, args):
    batch_data = []
    for problem in questions:
        if problem["task_id"] in processed_task_ids:
            continue
        user_message = create_user_message(problem["prompt"], problem["corrupted_solution"], args.is_explicit, test_code=problem["test_code"] if args.include_test_cases else None)
        batch_data.append({"message": user_message, "problem": problem, "user_message": user_message})

    return batch_data


def _process_response(resp_dict, data, args):
    final_answer = resp_dict.get("final_answer", "")
    error_msg = resp_dict.get("error")

    if error_msg or not final_answer:
        print(f"Skipping {data['problem']['task_id']} due to error: {error_msg or 'empty answer'}")
        return None

    prob = data["problem"]
    result = {
        "task_id": prob["task_id"],
        "solution": extract_code_from_response(final_answer),
        "llm_response": final_answer,
        "llm_reasoning": resp_dict.get("reasoning", ""),
        "corrupted_solution": prob["corrupted_solution"],
        "prompt": data["user_message"],
        "canonical_solution": prob["canonical_solution"],
        "test_code": prob["test_code"],
    }

    if args.store_token_info:
        token_info = resp_dict.get("token_usage", {})
        if token_info:
            result["token_usage"] = token_info

    return result


def main():
    args = get_args()
    if args.generic:
        assert args.num_shots == 0, "Generic mode is only allowed with num_shots=0"  # because few shot with generic is equivalent to few shot + explicit

    model = get_model(args.model, args.is_reasoning)

    questions = []
    with open(args.questions_path, "r") as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    shots_string, questions = generate_shots(questions, args.num_shots, args.include_test_cases)

    system_prompt = get_system_prompt_with_shots(shots_string, args.is_explicit, args.generic)
    explicit_mode = "explicit" if args.is_explicit else "standard"
    test_cases_mode = "with test cases" if args.include_test_cases else "without test cases"
    if args.num_shots > 0:
        print(f"Using {args.num_shots} few-shot examples ({explicit_mode} prompts, {test_cases_mode}). Total {len(questions)} questions from original {len(questions) + args.num_shots} questions.")
    else:
        print(f"Using zero-shot approach ({explicit_mode} prompts, {test_cases_mode}). Total {len(questions)} questions.")

    output_file = _generate_output_path(args, questions)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving results to {output_file}")
    # Load existing results to avoid reprocessing
    processed_task_ids = set()
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    processed_task_ids.add(result["task_id"])
                    existing_results[result["task_id"]] = result

    if set(q["task_id"] for q in questions).issubset(processed_task_ids):
        print(f"All {len(questions)} tasks already processed. Re-extracting code from existing responses...")

        updated_results = [
            {**existing_results[q["task_id"]], "solution": extract_code_from_response(existing_results[q["task_id"]]["llm_response"])} for q in questions if q["task_id"] in existing_results
        ]

        # Compute similarity metrics for re-extracted results
        try:
            similarity_calc = SimilarityCalculator()
            eval_map = {}
            for rec in updated_results:
                task_id = rec.get("task_id")
                task_data = {
                    "canonical_solution": rec.get("canonical_solution", ""),
                    "corrupted_solution": rec.get("corrupted_solution", ""),
                    "solution": rec.get("solution", ""),
                }
                per_metrics = _process_similarity_metrics(task_data, True, similarity_calc)
                # Difference helper expects canonical_vs_corrupted and llm_vs_corrupted keys
                per_metrics["difference_with_canonical"] = _compute_difference_with_canonical(per_metrics)
                rec["similarity_metrics"] = per_metrics
                eval_map[task_id] = {
                    "canonical_solution": task_data["canonical_solution"],
                    "corrupted_solution": task_data["corrupted_solution"],
                    "solution": task_data["solution"],
                    "similarity_metrics": per_metrics,
                }

            avg_similarity = _calculate_corpus_metrics({"eval": eval_map}, True)
        except Exception as e:
            print(f"Warning: similarity metric computation failed during re-extraction: {e}")
            eval_map = {}
            avg_similarity = {}

        # Write updated per-sample results back to the main JSONL file
        with open(output_file, "w") as f:
            for result in updated_results:
                f.write(json.dumps(result) + "\n")

        # Persist a summary metrics file alongside the results
        try:
            results_dir = os.path.join(os.path.dirname(output_file), "results")
            os.makedirs(results_dir, exist_ok=True)
            summary_path = os.path.join(results_dir, os.path.basename(output_file).replace(".jsonl", "_reextract_eval_results.json"))
            summary_obj = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "eval": eval_map,
                "metrics": {
                    "eval_similarity": True,
                    "similarity_metrics": avg_similarity,
                },
            }
            with open(summary_path, "w") as fsum:
                json.dump(summary_obj, fsum, indent=2)
            print(f"Re-extraction similarity metrics saved to {summary_path}")
        except Exception as e:
            print(f"Warning: failed to save re-extraction similarity metrics: {e}")

        print(f"Updated {len(updated_results)} results with re-extracted code and computed similarity metrics")
        return

    print(f"Skipping {len(processed_task_ids)}/{len(questions)} tasks")

    try:
        batch_data = _prepare_batch_data(questions, processed_task_ids, args)
        if not batch_data:
            print("No new questions to process")
            return

        responses = model.generate_responses(system_prompt, [data["message"] for data in batch_data])

        with open(output_file, "a") as f:
            for resp_dict, data in zip(responses, batch_data):
                result = _process_response(resp_dict, data, args)
                if result:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

    print(f"Results saved to {output_file}")

    # model.print_usage()


if __name__ == "__main__":
    main()
