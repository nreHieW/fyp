from models.base import BaseModel
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# we need to use streaming here because some of the reasoning models return too long an output and it fails.


class OpenRouterModel(BaseModel):
    def __init__(self, is_reasoning: bool, model_name: str, api_key: str | None = None):
        super().__init__()
        self.is_reasoning = is_reasoning
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")

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

    def generate_response(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        try:
            headers = self._get_headers()

            payload = {
                "model": self.model_name,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "max_tokens": 64000,
            }

            # Use streaming for reasoning models
            if self.is_reasoning:
                payload["stream"] = True
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, stream=True)
                response.raise_for_status()

                final_answer, reasoning = self._parse_streaming_response(response)

                # Note: Streaming responses don't always include usage data
                # We'll estimate or rely on the final response if available
                self.total_requests += 1

                # For streaming responses, we can't get reliable token usage
                return {
                    "reasoning": reasoning,
                    "final_answer": final_answer,
                    "token_usage": {
                        "prompt_tokens": 0,  # Not available in streaming
                        "completion_tokens": 0,  # Not available in streaming
                        "reasoning_tokens": 0,  # Not available in streaming
                        "total_tokens": 0,  # Not available in streaming
                    },
                }

            else:
                # Non-streaming mode
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()

                final_answer = response_data["choices"][0]["message"]["content"]
                reasoning = ""

                if self.is_reasoning and "reasoning" in response_data["choices"][0]["message"]:
                    reasoning = response_data["choices"][0]["message"]["reasoning"]

                # Prepare the result dictionary
                result = {"reasoning": reasoning, "final_answer": final_answer}

                # Track token usage and include in response when available
                if "usage" in response_data:
                    usage = response_data["usage"]
                    self.total_input_tokens += usage.get("prompt_tokens", 0)
                    self.total_output_tokens += usage.get("completion_tokens", 0)
                    self.total_reasoning_tokens += usage.get("reasoning_tokens", 0)
                    self.total_requests += 1

                    # Always include token usage when available
                    result["token_usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "reasoning_tokens": usage.get("reasoning_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                return result

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
            result = self._generate_response_without_counting(system_prompt, prompt)
            return index, result

        # Use ThreadPoolExecutor for concurrent processing
        max_workers = min(len(user_prompts), 10)  # Limit concurrent requests
        results = [None] * len(user_prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_prompt_with_index, (i, prompt)): i for i, prompt in enumerate(user_prompts)}

            # Process completed requests with progress bar
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

        # Aggregate token usage after all requests complete (no lock needed!)
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

    def _generate_response_without_counting(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        """Generate response without updating token counters (for thread safety)."""
        try:
            headers = self._get_headers()

            payload = {"model": self.model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}

            # Use streaming for reasoning models
            if self.is_reasoning:
                payload["stream"] = True
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, stream=True)
                response.raise_for_status()

                final_answer, reasoning = self._parse_streaming_response(response)

                # For streaming responses, we can't get reliable token usage
                return {
                    "reasoning": reasoning,
                    "final_answer": final_answer,
                    "token_usage": {
                        "prompt_tokens": 0,  # Not available in streaming
                        "completion_tokens": 0,  # Not available in streaming
                        "reasoning_tokens": 0,  # Not available in streaming
                        "total_tokens": 0,  # Not available in streaming
                    },
                }

            else:
                # Non-streaming mode
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()

                final_answer = response_data["choices"][0]["message"]["content"]
                reasoning = ""

                if self.is_reasoning and "reasoning" in response_data["choices"][0]["message"]:
                    reasoning = response_data["choices"][0]["message"]["reasoning"]

                # Prepare the result dictionary
                result = {"reasoning": reasoning, "final_answer": final_answer}

                # Include token usage when available (but don't update counters)
                if "usage" in response_data:
                    usage = response_data["usage"]
                    result["token_usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "reasoning_tokens": usage.get("reasoning_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                return result

        except Exception as e:
            return {"reasoning": "", "final_answer": "", "error": str(e)}
