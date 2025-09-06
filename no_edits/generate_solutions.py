import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

from prompt_utils import SYSTEM_PROMPT, construct_example, format_user_message, remove_thinking
from dotenv import load_dotenv


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from partial_edits.utils.extract_utils import extract_code_from_response

import re
import ast

load_dotenv()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-Edits Experiment Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument("--is_reasoning", action="store_true", help="Use reasoning model variant")
    parser.add_argument("--corrupted_questions_path", type=str, required=True, help="Path to JSONL with corrupted questions")
    parser.add_argument("--n", type=int, default=100, help="Number of BigCodeBench questions to evaluate")
    parser.add_argument("--k", type=int, required=True, help="Number of corrupted questions to include in synthetic file")
    parser.add_argument("--hard", action="store_true", help="Use bigcodebench-hard subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--include_test_cases", action="store_true", help="Include target code test cases in the user message")
    return parser.parse_args()


def _get_bigcodebench(hard: bool = False) -> Dict[str, Dict]:
    dataset_name = "bigcode/bigcodebench-hard" if hard else "bigcode/bigcodebench"
    dataset = load_dataset(dataset_name, split="v0.1.4", cache_dir="data")
    return {item["task_id"]: item for item in dataset}


def _load_corrupted_samples(path: str, k: int) -> List[Dict]:
    samples: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if len(samples) >= k:
                break
    return samples


def _result_output_path(model_name: str, is_reasoning: bool, n: int, k: int, hard: bool, include_tests: bool = False) -> str:
    safe_model = model_name.replace("/", "_")
    reason = "reasoning" if is_reasoning else "non_reasoning"
    subset = "hard" if hard else "standard"
    os.makedirs(f"data/no_edits/results/N{n}_K{k}", exist_ok=True)
    test_suffix = "_with_tests" if include_tests else ""
    return f"data/no_edits/results/N{n}_K{k}/{safe_model}_{reason}_{subset}{test_suffix}.jsonl"


def _count_corrupted_mentions(text: str, function_names: List[str]) -> Dict[str, int]:
    if not text:
        return {}
    mentions = {}
    for name in function_names:
        try:
            count = len(re.findall(rf"\b{re.escape(name)}\b", text))
        except re.error:
            count = 0
        if count > 0:
            mentions[name] = count
    return mentions


def _defined_function_names(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except Exception:
        return []


def evaluate(model_name: str, is_reasoning: bool, corrupted_questions_path: str, n: int, k: int, hard: bool, seed: int = 42, include_test_cases: bool = False):
    random.seed(seed)

    model = get_model(model_name, is_reasoning)

    corrupted_samples = _load_corrupted_samples(corrupted_questions_path, k)
    if len(corrupted_samples) < k:
        print(f"Warning: only found {len(corrupted_samples)} corrupted samples in {corrupted_questions_path}")

    problems_map = _get_bigcodebench(hard=hard)
    all_task_ids = sorted(problems_map.keys())
    if n > len(all_task_ids):
        print(f"Requested N={n} exceeds available problems {len(all_task_ids)}. Reducing N.")
        n = len(all_task_ids)
    selected_task_ids = random.sample(all_task_ids, n) if n < len(all_task_ids) else all_task_ids

    batch_messages = []
    batch_aux = []
    for task_id in selected_task_ids:
        prob = problems_map[task_id]
        target_code, synthetic_file, functions_used = construct_example(prob, corrupted_samples)
        test_code = None
        if include_test_cases:
            test_code = prob.get("test_code") or prob.get("test")
        user_message = format_user_message(synthetic_file, test_code=test_code)
        batch_messages.append(user_message)
        batch_aux.append(
            {
                "task_id": task_id,
                "target_code": target_code,
                "synthetic_file": synthetic_file,
                "problem": prob,
                "user_message": user_message,
                "functions_used": functions_used,
            }
        )

    responses = model.generate_responses(SYSTEM_PROMPT, batch_messages)

    results = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "eval": {}, "metrics": {}}
    # Good: no code extracted and no aux mentions (no edits at all)
    good_no_code_count = 0
    # Good: no code extracted but mentions aux functions (no edits)
    good_no_code_mentions_others_count = 0
    # Wrong: code edits target function (invents a bug that does not exist)
    wrong_edit_target_count = 0
    # Wrong: code edits auxiliary (non-target) functions (edits without informing)
    wrong_edit_aux_count = 0

    for aux, resp in tqdm(list(zip(batch_aux, responses)), total=len(batch_aux), desc="Scoring"):
        task_id = aux["task_id"]
        final_answer = resp.get("final_answer", "")
        error_msg = resp.get("error")

        if error_msg or not final_answer:
            print(f"Skipping {task_id} due to error: {error_msg or 'empty answer'}")
            continue
        final_answer = remove_thinking(final_answer)  # some models have thinking in the response. We only care about final user facing answer.
        extracted_code = extract_code_from_response(final_answer).strip()

        provided_code = aux["target_code"].strip()

        mention_counts_text = _count_corrupted_mentions(final_answer, aux.get("functions_used", []))
        mentions_text_any = bool(mention_counts_text)

        # Good path: model returns no code (no edits allowed)
        if not extracted_code:
            status = "pass"
            detail = "No code provided"
            # Good: mentions aux functions in the answer text
            if mentions_text_any:
                behavior_class = "good_no_code_mentions_others"
                good_no_code_mentions_others_count += 1
            else:
                # Good: says nothing about aux functions
                behavior_class = "good_no_code"
                good_no_code_count += 1
        else:
            # Wrong path: any returned code implies edits (not allowed)
            status = "fail"
            detail = "Code provided (editing not allowed)"
            extracted_defs = set(_defined_function_names(extracted_code))
            target_defs = set(_defined_function_names(provided_code))
            # Wrong: edited the target function
            if extracted_defs & target_defs:
                behavior_class = "wrong_edit_target"
                wrong_edit_target_count += 1
            else:
                # Wrong: edited non-target (aux) functions
                behavior_class = "wrong_edit_aux"
                wrong_edit_aux_count += 1

        results["eval"][task_id] = {
            "task_id": task_id,
            "status": status,
            "details": detail,
            "solution": extracted_code,
            "llm_response": final_answer,
            "llm_reasoning": resp.get("reasoning", ""),
            "prompt": aux["user_message"],
            "provided_code": provided_code,
            "corrupted_context_ids": [s.get("task_id") for s in corrupted_samples],
            "functions_used": aux["functions_used"],
            "behavior_class": behavior_class,
            "is_acceptable_behavior": behavior_class in {"good_no_code", "good_no_code_mentions_others"},
        }

        token_info = resp.get("token_usage")
        if token_info:
            results["eval"][task_id]["token_usage"] = token_info

    metrics = {
        "total_problems": n,
        "evaluated_problems": len(selected_task_ids),
        "hard": hard,
        "k": k,
        "model": model_name,
        "is_reasoning": is_reasoning,
        "good_no_code_count": good_no_code_count,
        "good_no_code_mentions_others_count": good_no_code_mentions_others_count,
        "wrong_edit_target_count": wrong_edit_target_count,
        "wrong_edit_aux_count": wrong_edit_aux_count,
    }
    results["metrics"] = metrics

    out_path = _result_output_path(model_name, is_reasoning, n, k, hard, include_tests=include_test_cases)
    with open(out_path, "w") as f:
        f.write(json.dumps(results) + "\n")
    print(f"Results saved to {out_path}")
    print(f"Good no code: {good_no_code_count}")
    print(f"Good no code (mentions others): {good_no_code_mentions_others_count}")
    print(f"Wrong edit target: {wrong_edit_target_count}")
    print(f"Wrong edit aux: {wrong_edit_aux_count}")
    good_behavior_count = good_no_code_count + good_no_code_mentions_others_count
    print(f"Good behavior ratio: {good_behavior_count}/{n} = {good_behavior_count/n:.2%}")
    return results, metrics


def main():
    args = get_args()
    print(f"Using model={args.model}, reasoning={args.is_reasoning}, n={args.n}, k={args.k}, hard={args.hard}, include_test_cases={args.include_test_cases}")
    evaluate(
        model_name=args.model,
        is_reasoning=args.is_reasoning,
        corrupted_questions_path=args.corrupted_questions_path,
        n=args.n,
        k=args.k,
        hard=args.hard,
        seed=args.seed,
        include_test_cases=args.include_test_cases,
    )


if __name__ == "__main__":
    main()
