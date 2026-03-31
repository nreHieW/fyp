# @misc{deepcoder2025,
#   title={DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level},
#   author={Michael Luo and Sijun Tan and Roy Huang and Ameen Patel and Alpay Ariyak and Qingyang Wu and Xiaoxiang Shi and Rachel Xin and Colin Cai and Maurice Weber and Ce Zhang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
#   howpublished={\url{https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51}},
#   note={Notion Blog},
#   year={2025}

import asyncio
import json
import os
import random
from typing import Callable, Dict, List
import concurrent.futures
import verifiers as vf
from datasets import load_dataset
from deepcoder_utils.legacy.deepcoder_genesys import extract_code_from_model
from deepcoder_utils.local_verify import verify_deepcoder_local
from partial_edits_utils.prompt_utils import SYSTEM_PROMPT, create_user_message
from partial_edits_utils.similarity_utils import get_cognitive_complexity_similarity, get_levenshtein_distance

LOWEST_SCORE = -0.2

class CodeBlockParser:
    """Parser to extract code from model responses after ThinkParser processing."""

    def __init__(self, extract_fn: Callable[[str], str] = extract_code_from_model, **kwargs):
        super().__init__(**kwargs)
        self.extract_fn = extract_fn

    def parse(self, text: str) -> str:
        # allow non-thinking responses for debugging after verifiers>0.1.3
        # TODO: revert for training with reasoners
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return self.extract_fn(text.strip())
        # return super().parse(text)


class DeepCoderEnv(vf.SingleTurnEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def setup_state(self, state, **kwargs):
        # No sandbox setup; evaluation runs locally
        return state

    def process_env_results_vllm(self, *args, **kwargs):
        processed_outputs = super().process_env_results_vllm(*args, **kwargs)
        # for exceptions not caused by generated code (e.g., infra failures), zero out completion mask
        for i, reward in enumerate(processed_outputs.rewards):
            if reward is None:
                processed_outputs.completion_mask[i] = [0] * len(processed_outputs.completion_ids[i])
        return processed_outputs


class DeepCoderRubric:
    def __init__(
        self,
        parser: CodeBlockParser,
        timeout_per_test: int = 20,
        max_tests: int = 2,
        similarity_metric: str = "levenshtein",
        similarity_weight: float = 1.0,
        execution_weight: float = 0.1,
    ):
        self.parser = parser
        self.timeout_per_test = timeout_per_test
        self.max_tests = max_tests
        self.similarity_metric = similarity_metric
        self.similarity_weight = similarity_weight
        self.execution_weight = execution_weight

    def _similarity_score(self, canonical: str, corrupted: str, completion: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        if self.similarity_metric in {"levenshtein", "both"}:
            baseline_distance = get_levenshtein_distance(canonical, corrupted, normalize=True, ignore_comments=True)
            completion_distance = get_levenshtein_distance(completion, corrupted, normalize=True, ignore_comments=True)
            scores["levenshtein"] = completion_distance - baseline_distance

        if self.similarity_metric in {"cognitive_complexity", "both"}:
            baseline_similarity = get_cognitive_complexity_similarity(canonical, corrupted)
            completion_similarity = get_cognitive_complexity_similarity(completion, corrupted)
            scores["cognitive_complexity"] = completion_similarity - baseline_similarity

        return scores

    async def deepcoder_reward_func(
        self,
        completion: str,
        info: dict,
    ) -> tuple[float, Dict[str, float]]:
        """Execute code against test cases using deepcoder verification system."""
        parsed_completion = self.parser.parse(completion[0]["content"])
        loop = asyncio.get_running_loop()
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        try:
            try:
                execution_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        pool,
                        verify_deepcoder_local,
                        parsed_completion,
                        info,
                        self.timeout_per_test,
                        self.max_tests,
                    ),
                    timeout=self.timeout_per_test * self.max_tests,
                )
            except asyncio.TimeoutError:
                execution_result = 0
        finally:
            # Shutdown executor without waiting for stuck processes
            # This ensures termination even if verify_deepcoder_local is stuck in an infinite loop
            pool.shutdown(wait=False)

        if execution_result == 0:
            return 0, {}
        elif execution_result == 1:
            similarity_score = self._similarity_score(
                info["canonical_solution"],
                info["corrupted_solution"],
                parsed_completion,
            )
            return execution_result, similarity_score
        else:
            raise ValueError(f"Invalid execution_result: {execution_result}")

    async def score_rollout(
        self,
        completion: str,
        info: dict | None = None,
    ) -> float:
        execution_reward, similarity_reward = await self.deepcoder_reward_func(
            completion=completion,
            info=info,
        )
        if execution_reward == 0:
            return LOWEST_SCORE

        similarity_reward = {k: -v for k, v in similarity_reward.items()}
        reward = self.execution_weight * execution_reward + self.similarity_weight * sum(similarity_reward.values())
        return reward


def _process_test(test: str) -> dict:
    tests = json.loads(test)
    if isinstance(tests, dict):
        tests = [tests]

    for test in tests:
        if "inputs" in test:
            test["input"] = test.pop("inputs")

        if "outputs" in test:
            test["output"] = test.pop("outputs")

    return json.dumps(tests)


def load_environment(
    timeout_per_test: int = 60,
    max_tests: int = 2,
    ds_num_proc: int = max(1, os.cpu_count() // 2),
    seed: int = 42,
    env_type: str = "both",
    similarity_metric: str = "levenshtein",
    similarity_weight: float = 1.0,
    execution_weight: float = 0.1,
    sort_by_difficulty: bool = False,
    **kwargs,
) -> vf.Environment:
    """Load DeepCoder environment for coding problems with executable verification."""
    assert env_type in ["non_ood", "ood", "both"]
    print(f"Env_type: {env_type}, Similarity_metric: {similarity_metric}, Similarity_weight: {similarity_weight}, Execution_weight: {execution_weight}")
    random.seed(seed)
    train_dataset = load_dataset("nreHieW/DeepCoder-Partial-Edits" + ("-" + env_type if env_type in ["both", "ood"] else "") + "-filtered", split="train").shuffle(seed=seed)

    # Ensure corrupted examples only
    train_dataset = train_dataset.filter(lambda x: x["corrupted_answer"] is not None and len(x["corrupted_answer"]) < 2000)

    if sort_by_difficulty:
        train_dataset = train_dataset.map(
            lambda example: {"_applied_mutations_len": len(example.get("applied_mutations", [])) if isinstance(example.get("applied_mutations"), list) else 0},
            num_proc=ds_num_proc,
        )
        train_dataset = train_dataset.sort("_applied_mutations_len")
        train_dataset = train_dataset.remove_columns(["_applied_mutations_len"])
    train_dataset = train_dataset.map(
        lambda x: {
            "question": create_user_message(x["problem_spec"], x["corrupted_answer"]),
            "answer": x["correct_answer"],
            "info": {
                "dataset_type": "primeintellect",
                "tests": _process_test(x["tests"]),
                "canonical_solution": x["correct_answer"],
                "corrupted_solution": x["corrupted_answer"],
            },
            "task": "deepcoder",
        },
        num_proc=ds_num_proc,
    )

    parser = CodeBlockParser()

    rubric = DeepCoderRubric(
        parser=parser,
        timeout_per_test=timeout_per_test,
        max_tests=max_tests,
        similarity_metric=similarity_metric,
        similarity_weight=similarity_weight,
        execution_weight=execution_weight,
    )

    vf_env = DeepCoderEnv(
        dataset=train_dataset,
        parser=parser,
        rubric=vf.Rubric(
            funcs=[rubric.score_rollout],
            weights=[1.0],
        ),
        system_prompt=SYSTEM_PROMPT,
    )
    return vf_env
