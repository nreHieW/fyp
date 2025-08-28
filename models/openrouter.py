from models.base import BaseModel
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

# Use normal requests by default, fallback to streaming if they fail due to long responses.


class OpenRouterModel(BaseModel):
    def __init__(self, is_reasoning: bool, model_name: str, api_key: str | None = None):
        super().__init__()
        self.is_reasoning = is_reasoning
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")

        self.thinking_effort = None
        if is_reasoning:
            thinking_pattern = r"_(low|medium|high)$"
            match = re.search(thinking_pattern, model_name)
            if match:
                thinking_level = match.group(1)
                self.thinking_effort = thinking_level
                self.model_name = re.sub(thinking_pattern, "", model_name)

    def _get_headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _parse_streaming_response(self, response):
        """Parse streaming response and return content and reasoning."""
        content_parts = []
        reasoning = ""

        buffer = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                buffer += chunk
                while True:
                    try:
                        # Find the next complete SSE line
                        line_end = buffer.find("\n")
                        if line_end == -1:
                            break

                        line = buffer[:line_end].strip()
                        buffer = buffer[line_end + 1 :]

                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break

                            try:
                                data_obj = json.loads(data)
                                delta = data_obj["choices"][0]["delta"]

                                # Extract content
                                content = delta.get("content")
                                if content:
                                    content_parts.append(content)

                                # Extract reasoning if available
                                if self.is_reasoning and "reasoning" in delta:
                                    reasoning += delta["reasoning"]

                            except json.JSONDecodeError:
                                pass
                    except Exception:
                        break

        return "".join(content_parts), reasoning

    def generate_response(self, system_prompt: str, user_prompt: str, update_counters: bool = True) -> dict[str, str]:
        try:
            headers = self._get_headers()

            payload = {
                "model": self.model_name,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                # "max_tokens": 128000,
            }
            if self.is_reasoning:
                if self.thinking_effort:
                    payload["reasoning"] = {"effort": self.thinking_effort, "enabled": True}
                else:
                    payload["reasoning"] = {"enabled": True}

            if not self.is_reasoning:
                payload["reasoning"] = {"enabled": False}

            try:
                # First try normal (non-streaming) mode
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()

                final_answer = response_data["choices"][0]["message"]["content"]
                finish_reason = response_data["choices"][0]["finish_reason"]
                if finish_reason not in ["stop", "length"]:
                    print(finish_reason)
                    raise Exception(f"Finish reason: {finish_reason}")
                reasoning = ""

                if "reasoning" in response_data["choices"][0]["message"]:
                    reasoning = response_data["choices"][0]["message"]["reasoning"]

                result = {"reasoning": reasoning, "final_answer": final_answer}

                # Track token usage and include in response when available
                if "usage" in response_data:
                    usage = response_data["usage"]
                    if update_counters:
                        self.total_input_tokens += usage.get("prompt_tokens", 0)
                        self.total_output_tokens += usage.get("completion_tokens", 0)
                        self.total_reasoning_tokens += usage.get("reasoning_tokens", 0)
                        self.total_requests += 1

                    result["token_usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "reasoning_tokens": usage.get("reasoning_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                return result

            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                payload["stream"] = True
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, stream=True)
                response.raise_for_status()

                final_answer, reasoning = self._parse_streaming_response(response)
                if update_counters:
                    self.total_requests += 1

                return {
                    "reasoning": reasoning,
                    "final_answer": final_answer,
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "reasoning_tokens": 0,
                        "total_tokens": 0,
                    },
                }

        except Exception as e:
            return {"reasoning": "", "final_answer": "", "error": str(e)}

    def generate_responses(self, system_prompt: str, user_prompts: list[str]):
        """Generate responses for a batch of user prompts using concurrent processing.

        Uses ThreadPoolExecutor to process multiple prompts in parallel for better performance.
        """
        if not user_prompts:
            return []

        def process_prompt_with_index(args):
            index, prompt = args
            result = self.generate_response(system_prompt, prompt, update_counters=False)
            return index, result

        max_workers = min(len(user_prompts), 3)
        results = [None] * len(user_prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(process_prompt_with_index, (i, prompt)): i for i, prompt in enumerate(user_prompts)}

            with tqdm(total=len(user_prompts), desc="Processing prompts") as pbar:
                for future in as_completed(future_to_index):
                    try:
                        index, result = future.result()
                        results[index] = result
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing prompt {index}: {e}")
                        index = future_to_index[future]
                        results[index] = {"reasoning": "", "final_answer": "", "error": str(e)}
                        pbar.update(1)

        # Update token counters after all requests complete
        successful_requests = 0
        for result in results:
            if result and "token_usage" in result and "error" not in result:
                usage = result["token_usage"]
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                self.total_reasoning_tokens += usage.get("reasoning_tokens", 0)
                successful_requests += 1

        self.total_requests += successful_requests

        return results
