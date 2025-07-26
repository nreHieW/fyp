from models.base import BaseModel
from openai import OpenAI
import os
import re


class GoogleModel(BaseModel):
    def __init__(self, is_reasoning: bool, model_name: str, api_key: str | None = None):
        super().__init__()
        self.is_reasoning = is_reasoning

        self.thinking_budget = None
        self.clean_model_name = model_name

        if is_reasoning:
            thinking_pattern = r"_(low|medium|high)$"
            match = re.search(thinking_pattern, model_name)
            if match:
                thinking_level = match.group(1)
                budget_map = {"low": 1024, "medium": 8192, "high": 24576}
                self.thinking_budget = budget_map.get(thinking_level, "none")
                self.clean_model_name = re.sub(thinking_pattern, "", model_name)

        self.client = OpenAI(api_key=api_key or os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    def generate_response(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        try:
            request_params = {
                "model": self.clean_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            if self.is_reasoning:
                thinking_config = {"include_thoughts": True}
                if self.thinking_budget:
                    thinking_config["thinking_budget"] = self.thinking_budget

                request_params["extra_body"] = {"extra_body": {"google": {"thinking_config": thinking_config}}}
            else:
                request_params["reasoning_effort"] = "none"

            response = self.client.chat.completions.create(**request_params)

            final_answer = response.choices[0].message.content
            reasoning = ""

            if self.is_reasoning and hasattr(response.choices[0].message, "reasoning") and response.choices[0].message.reasoning:
                reasoning = response.choices[0].message.reasoning

            result = {"reasoning": reasoning, "final_answer": final_answer}
            if hasattr(response, "usage") and response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.total_tokens - response.usage.prompt_tokens  # account for reasoning tokens
                # Track reasoning tokens if available (for models like o1)
                if hasattr(response.usage, "reasoning_tokens") and response.usage.reasoning_tokens:
                    reasoning_tokens = response.usage.reasoning_tokens
                    self.total_reasoning_tokens += reasoning_tokens
                else:
                    reasoning_tokens = response.usage.total_tokens - response.usage.completion_tokens - response.usage.prompt_tokens
                    self.total_reasoning_tokens += reasoning_tokens
                self.total_requests += 1

                result["token_usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return result
        except Exception as e:
            return {"reasoning": "", "final_answer": "", "error": str(e)}

    def generate_responses(self, system_prompt: str, user_prompts: list[str]):
        """Generate responses for a batch of user prompts.

        Google API doesn't support batch processing, so we use sequential processing.
        """
        return super().generate_responses(system_prompt, user_prompts)
