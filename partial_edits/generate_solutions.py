import json
import os
import argparse
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.extract_utils import *
from utils.prompts_utils import generate_shots, create_user_message, get_system_prompt_with_shots
from models import get_model

load_dotenv(override=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to use for evaluation")
    parser.add_argument("--is_reasoning", action="store_true", help="Whether to use reasoning model")
    parser.add_argument("--questions_path", type=str, required=True, help="Path to the JSONL file containing questions")
    parser.add_argument("--num_shots", type=int, default=0, help="Number of shots to use for few-shot learning")
    parser.add_argument("--is_explicit", action="store_true", help="Whether to use explicit prompts with more detailed instructions")
    parser.add_argument("--include_test_cases", action="store_true", help="Whether to include test cases in the user message")
    parser.add_argument("--store_token_info", action="store_true", help="Whether to store token usage information on a per-sample basis")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # Always create the model with batch support enabled
    model = get_model(args.model, args.is_reasoning)

    # Load questions from JSONL file
    questions = []
    with open(args.questions_path, "r") as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    shots_string, questions = generate_shots(questions, args.num_shots, args.include_test_cases)

    system_prompt = get_system_prompt_with_shots(shots_string, args.num_shots, args.is_explicit)
    explicit_mode = "explicit" if args.is_explicit else "standard"
    test_cases_mode = "with test cases" if args.include_test_cases else "without test cases"
    if args.num_shots > 0:
        print(f"Using {args.num_shots} few-shot examples ({explicit_mode} prompts, {test_cases_mode}). Total {len(questions)} questions from original {len(questions) + args.num_shots} questions.")
    else:
        print(f"Using zero-shot approach ({explicit_mode} prompts, {test_cases_mode}). Total {len(questions)} questions.")

    model_name = args.model.split("/")[-1]
    os.makedirs("data/code_edits", exist_ok=True)

    # Determine file suffix based on explicit mode
    explicit_suffix = "_explicit" if args.is_explicit else ""
    test_cases_suffix = "_with_tests" if args.include_test_cases else ""

    if args.num_shots > 0:
        os.makedirs(f"data/code_edits/{args.num_shots}_shot{explicit_suffix}{test_cases_suffix}", exist_ok=True)
        output_file = f"data/code_edits/{args.num_shots}_shot{explicit_suffix}{test_cases_suffix}/results_{model_name}_{'reasoning' if args.is_reasoning else 'non_reasoning'}_{len(questions)}.jsonl"
    else:
        output_file = f"data/code_edits/results_{model_name}_{'reasoning' if args.is_reasoning else 'non_reasoning'}{explicit_suffix}{test_cases_suffix}_{len(questions)}.jsonl"
    print(f"Saving results to {output_file}")
    # Load existing results to avoid reprocessing
    processed_task_ids = set()
    existing_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line.strip())
                        processed_task_ids.add(result["task_id"])
                        existing_results[result["task_id"]] = result
        except json.JSONDecodeError:
            print(f"Error loading {output_file}, starting fresh")

    # Check if all questions are already processed
    all_task_ids = set(q["task_id"] for q in questions)
    if all_task_ids.issubset(processed_task_ids):
        print(f"All {len(questions)} tasks already processed. Re-extracting code from existing responses...")

        # Re-extract code from existing responses and update results
        updated_results = []
        for question in questions:
            task_id = question["task_id"]
            if task_id in existing_results:
                result = existing_results[task_id].copy()
                # Re-extract the code using the updated extraction function
                llm_solution = extract_code_from_response(result["llm_response"])

                result["solution"] = llm_solution
                updated_results.append(result)

        # Write updated results back to file
        with open(output_file, "w") as f:
            for result in updated_results:
                f.write(json.dumps(result) + "\n")

        print(f"Updated {len(updated_results)} results with re-extracted code")
        model.print_usage()
        return

    print(f"Skipping {len(processed_task_ids)}/{len(questions)} tasks")

    try:
        with open(output_file, "a") as f:
            batch_messages = []
            batch_problems = []
            batch_user_messages = []

            for problem in questions:
                if problem["task_id"] in processed_task_ids:
                    continue

                problem_statement = problem["prompt"]
                corrupted_solution = problem["corrupted_solution"]

                user_message = create_user_message(problem_statement, corrupted_solution, args.is_explicit, test_code=problem["test_code"] if args.include_test_cases else None)

                batch_messages.append(user_message)
                batch_problems.append(problem)
                batch_user_messages.append(user_message)

            responses = model.generate_responses(system_prompt, batch_messages)

            for resp_dict, prob, user_msg in zip(responses, batch_problems, batch_user_messages):
                final_answer = resp_dict.get("final_answer", "")
                error_msg = resp_dict.get("error")

                if error_msg or not final_answer:
                    print(f"Skipping {prob['task_id']} due to error: {error_msg or 'empty answer'}")
                    continue

                reasoning = resp_dict.get("reasoning", "")
                llm_solution = extract_code_from_response(final_answer)

                result_entry = {
                    "task_id": prob["task_id"],
                    "solution": llm_solution,
                    "llm_response": final_answer,
                    "llm_reasoning": reasoning,
                    "corrupted_solution": prob["corrupted_solution"],
                    "prompt": user_msg,
                    "canonical_solution": prob["canonical_solution"],
                    "test_code": prob["test_code"],
                }

                if args.store_token_info:
                    token_info = resp_dict.get("token_usage", {})
                    if token_info:
                        result_entry["token_usage"] = token_info

                f.write(json.dumps(result_entry) + "\n")
                f.flush()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

    print(f"Results saved to {output_file}")

    model.print_usage()


if __name__ == "__main__":
    main()
