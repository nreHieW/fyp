from abc import ABC, abstractmethod

from tqdm import tqdm


class BaseModel(ABC):
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_requests = 0

    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        pass

    def generate_responses(self, system_prompt: str, user_prompts: list[str]):
        """Generate responses for a batch of user prompts.

        The default implementation simply calls `generate_response` sequentially. Models that
        support true batch inference should override this method.

        Args:
            system_prompt: The system prompt to use
            user_prompts: List of user prompts to process
            batch_job_id: Optional existing batch job ID to retrieve results from
        """
        return [self.generate_response(system_prompt, p) for p in tqdm(user_prompts)]

    def print_usage(self):
        print(f"\n--- Token Usage Statistics ---")
        print(f"Total Requests: {self.total_requests}")
        print(f"Total Input Tokens: {self.total_input_tokens:,}")
        print(f"Total Output Tokens: {self.total_output_tokens:,}")
        print(f"Total Reasoning Tokens: {self.total_reasoning_tokens:,}")
        print(f"Total Tokens: {self.total_input_tokens + self.total_output_tokens + self.total_reasoning_tokens:,}")
        if self.total_requests > 0:
            print(f"Average Input Tokens per Request: {self.total_input_tokens / self.total_requests:.1f}")
            print(f"Average Output Tokens per Request: {self.total_output_tokens / self.total_requests:.1f}")
            print(f"Average Reasoning Tokens per Request: {self.total_reasoning_tokens / self.total_requests:.1f}")
        print(f"------------------------------\n")

    def reset_usage(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_requests = 0
