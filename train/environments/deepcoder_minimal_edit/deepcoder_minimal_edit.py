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
from typing import Callable, List

import verifiers as vf
from datasets import load_dataset
from deepcoder_utils.legacy.deepcoder_genesys import extract_code_from_model
from deepcoder_utils.local_verify import verify_deepcoder_local
from partial_edits_utils.prompt_utils import SYSTEM_PROMPT, create_user_message
from partial_edits_utils.similarity_utils import get_levenshtein_distance
from verifiers.types import ChatMessage, Info, Messages, RolloutScores, State, ProcessedOutputs


class CodeBlockParser(vf.ThinkParser):
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

    async def setup_state(self, state: State, **kwargs) -> State:
        # No sandbox setup; evaluation runs locally
        return state

    def process_env_results_vllm(self, *args, **kwargs) -> ProcessedOutputs:
        processed_outputs = super().process_env_results_vllm(*args, **kwargs)
        # for exceptions not caused by generated code (e.g., infra failures), zero out completion mask
        for i, reward in enumerate(processed_outputs.rewards):
            if reward is None:
                processed_outputs.completion_mask[i] = [0] * len(processed_outputs.completion_ids[i])
        return processed_outputs


class DeepCoderRubric(vf.Rubric):
    """
    Environment for DeepCoder coding problems with executable verification.

    Sets up prime-intellect sandbox for each rollout.

    Supports the following task types:
    - `primeintellect`
    """

    def __init__(self, parser: CodeBlockParser, timeout_per_test: int = 60, max_tests: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.parser = parser
        self.timeout_per_test = timeout_per_test
        self.max_tests = max_tests

    async def deepcoder_reward_func(
        self,
        completion: str | List[ChatMessage],
        info: dict,
        **kwargs,
    ) -> float:
        """Execute code against test cases using deepcoder verification system."""
        try:
            parsed_completion = self.parser.parse(completion[0]["content"])
            # Run local verification in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: verify_deepcoder_local(
                    completion=parsed_completion,
                    verification_info=info,
                    timeout_per_test=self.timeout_per_test,
                    max_tests=self.max_tests,
                ),
            )
            if result == 0:
                return 0.0  # if wrong result always 0 (prevent hacking of just not editing)
            elif result == 1:
                normalized_levenshtein_distance = get_levenshtein_distance(info["canonical_solution"], parsed_completion, normalize=True)
                weight = 0.5
                normalized_levenshtein_distance *= weight
                return 1.0 - normalized_levenshtein_distance  # lower levenshtein distance is better
            else:
                raise ValueError(f"Invalid result: {result}")
        except Exception as e:
            print(f"Error in deepcoder verification: {repr(e)}")
            return 0.0

    async def score_rollouts(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        answers: List[str],
        states: List[State],
        tasks: List[str],
        infos: List[Info],
        **kwargs,
    ) -> RolloutScores:
        async def process_rollout(completion, info, state):
            # Run a single rollout locally
            return await self.deepcoder_reward_func(completion=completion, info=info, **kwargs)

        tasks = []
        for completion, info, state in zip(completions, infos, states):
            tasks.append(asyncio.create_task(process_rollout(completion, info, state)))
        rewards = await asyncio.gather(*tasks)
        return RolloutScores(reward=rewards, metrics={"deepcoder_reward_func": rewards})


def load_environment(
    timeout_per_test: int = 60,
    max_tests: int = 2,
    ds_num_proc: int = max(1, os.cpu_count() // 2),
    seed: int = 42,
    **kwargs,
) -> vf.Environment:
    """Load DeepCoder environment for coding problems with executable verification."""
    random.seed(seed)
    train_dataset = load_dataset("nreHieW/DeepCoder-Partial-Edits", split="train")
    # Ensure corrupted examples only
    train_dataset = train_dataset.filter(lambda x: x["corrupted_answer"] is not None)
    train_dataset = train_dataset.map(
        lambda x: {
            "question": create_user_message(x["problem_spec"], x["corrupted_answer"]),
            "answer": x["correct_answer"],
            "info": {
                "dataset_type": "primeintellect",
                "ground_truth": x["test_code"],
                "canonical_solution": x["correct_answer"],
                "corrupted_solution": x["corrupted_answer"],
            },
            "task": "deepcoder",
        },
        num_proc=ds_num_proc,
    )
    train_dataset = train_dataset.select(range(min(50, train_dataset.num_rows)))
    print(train_dataset)

    parser = CodeBlockParser()

    rubric = DeepCoderRubric(parser=parser, timeout_per_test=timeout_per_test, max_tests=max_tests)

    vf_env = DeepCoderEnv(
        dataset=train_dataset,
        parser=parser,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
    )
    return vf_env
