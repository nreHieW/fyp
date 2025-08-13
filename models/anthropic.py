from models.base import BaseModel
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.thinking_config_param import ThinkingConfigParam
import time
import os


class AnthropicModel(BaseModel):
    def __init__(self, is_reasoning: bool, model_name: str, api_key: str | None = None):
        super().__init__()
        self.is_reasoning = is_reasoning
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _parse_response_content(self, content_blocks) -> tuple[str, str]:
        """Parse response content to extract reasoning and final answer."""
        reasoning = ""
        final_answer = ""

        if isinstance(content_blocks, list):
            for block in content_blocks:
                if hasattr(block, "type"):
                    if block.type == "thinking":
                        reasoning = block.thinking
                    elif block.type == "text":
                        final_answer = block.text
        else:
            # Single text block fallback
            if hasattr(content_blocks, "text"):
                final_answer = content_blocks.text
            else:
                final_answer = str(content_blocks)

        return reasoning, final_answer

    def generate_response(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        try:
            # Prepare the message parameters
            message_params = {"model": self.model_name, "system": system_prompt, "messages": [{"role": "user", "content": user_prompt}]}

            # Add thinking parameter if reasoning is enabled
            if self.is_reasoning:
                message_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            response = self.client.messages.create(**message_params)

            reasoning, final_answer = self._parse_response_content(response.content)

            # Prepare the result dictionary
            result = {"reasoning": reasoning, "final_answer": final_answer}

            # Track token usage and include in response when available
            if hasattr(response, "usage") and response.usage:
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                self.total_requests += 1

                # Always include token usage when available
                result["token_usage"] = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "reasoning_tokens": 0,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }

            return result
        except Exception as e:
            return {"reasoning": "", "final_answer": "", "error": str(e)}

    def generate_responses(self, system_prompt: str, user_prompts: list[str]):
        """Generate responses using Anthropic's Message Batches API for efficient batch processing."""
        if len(user_prompts) == 1:
            # For single prompts, use regular API
            return [self.generate_response(system_prompt, user_prompts[0])]

        try:
            requests = []
            for idx, prompt in enumerate(user_prompts):

                if self.is_reasoning:
                    # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#how-to-use-extended-thinking
                    message_params = MessageCreateParamsNonStreaming(
                        model=self.model_name,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                        thinking={"type": "enabled", "budget_tokens": 10000},
                        max_tokens=16000,
                    )
                else:
                    message_params = MessageCreateParamsNonStreaming(model=self.model_name, system=system_prompt, messages=[{"role": "user", "content": prompt}], max_tokens=16000)

                requests.append(Request(custom_id=f"request-{idx}", params=message_params))

            # Create the batch
            message_batch = self.client.messages.batches.create(requests=requests)
            print(f"Created batch {message_batch.id} with {len(user_prompts)} requests")

            # Poll for completion
            while True:
                batch_status = self.client.messages.batches.retrieve(message_batch.id)
                if batch_status.processing_status == "ended":
                    break
                elif batch_status.processing_status in ["failed", "canceled", "expired"]:
                    print(f"Batch {message_batch.id} failed with status: {batch_status.processing_status}")
                    # Fallback to sequential processing
                    return super().generate_responses(system_prompt, user_prompts)

                print(f"Batch {message_batch.id} status: {batch_status.processing_status}")
                time.sleep(30)  # Poll every 30 seconds

            # Retrieve results
            results_map = {}
            for result in self.client.messages.batches.results(message_batch.id):
                custom_id = result.custom_id

                if result.result.type == "succeeded":
                    message = result.result.message
                    reasoning, final_answer = self._parse_response_content(message.content)

                    response_dict = {"reasoning": reasoning, "final_answer": final_answer}

                    if hasattr(message, "usage") and message.usage:
                        self.total_input_tokens += message.usage.input_tokens
                        self.total_output_tokens += message.usage.output_tokens
                        self.total_requests += 1

                        response_dict["token_usage"] = {
                            "prompt_tokens": message.usage.input_tokens,
                            "completion_tokens": message.usage.output_tokens,
                            "reasoning_tokens": 0,
                            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
                        }

                    results_map[custom_id] = response_dict

                elif result.result.type == "errored":
                    error_msg = str(result.result.error)
                    results_map[custom_id] = {"reasoning": "", "final_answer": "", "error": error_msg}

                elif result.result.type in ["canceled", "expired"]:
                    results_map[custom_id] = {"reasoning": "", "final_answer": "", "error": f"Request {result.result.type}"}

            # Build ordered results list
            ordered_results = []
            for idx in range(len(user_prompts)):
                custom_id = f"request-{idx}"
                ordered_results.append(results_map.get(custom_id, {"reasoning": "", "final_answer": "", "error": "Result missing"}))

            return ordered_results

        except Exception as e:
            print(f"Batch processing failed: {e}")
            # Fallback to sequential processing
            # return super().generate_responses(system_prompt, user_prompts)
            raise e
