from models.base import BaseModel
from openai import OpenAI
import time
import json
import tempfile, os
import re


class OpenAICompatibleModel(BaseModel):
    def __init__(self, is_reasoning: bool, model_name: str | None = None, base_url: str | None = None, api_key: str | None = None):
        super().__init__()
        self.is_reasoning = is_reasoning

        self.reasoning_effort_level = None
        self.clean_model_name = model_name

        # Only parse reasoning effort if reasoning is enabled
        if is_reasoning and model_name:
            reasoning_pattern = r"_(low|medium|high)$"
            match = re.search(reasoning_pattern, model_name)
            if match:
                self.reasoning_effort_level = match.group(1)
                self.clean_model_name = re.sub(reasoning_pattern, "", model_name)

        self.model_name = self.clean_model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_response(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        try:
            request_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            # Add reasoning_effort parameter if reasoning effort level is detected
            if self.reasoning_effort_level:
                request_params["reasoning_effort"] = self.reasoning_effort_level

            response = self.client.chat.completions.create(**request_params)

            final_answer = response.choices[0].message.content
            reasoning = ""

            if self.is_reasoning and hasattr(response.choices[0].message, "reasoning") and response.choices[0].message.reasoning:
                reasoning = response.choices[0].message.reasoning

            result = {"reasoning": reasoning, "final_answer": final_answer}

            if hasattr(response, "usage") and response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                output_tokens = response.usage.total_tokens - response.usage.prompt_tokens
                reasoning_tokens = output_tokens - response.usage.completion_tokens
                self.total_output_tokens += output_tokens
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

    def _parse_output_file(self, file_id: str) -> dict[str, dict[str, str]]:
        """Parse successful responses from batch output file."""
        resp_map = {}
        try:
            output_text = self.client.files.content(file_id).text
            for line in output_text.splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    custom_id = item.get("custom_id")
                    content = item["response"]["body"]["choices"][0]["message"]["content"]
                    reasoning = ""
                    if self.is_reasoning and "reasoning" in item["response"]["body"]["choices"][0]["message"]:
                        reasoning = item["response"]["body"]["choices"][0]["message"]["reasoning"]

                    result = {"reasoning": reasoning, "final_answer": content}

                    if "usage" in item["response"]["body"]:
                        usage = item["response"]["body"]["usage"]
                        self.total_input_tokens += usage.get("prompt_tokens", 0)
                        self.total_output_tokens += usage.get("completion_tokens", 0)
                        reasoning_tokens = usage.get("total_tokens", 0) - usage.get("prompt_tokens", 0) - usage.get("completion_tokens", 0)
                        self.total_reasoning_tokens += reasoning_tokens
                        self.total_requests += 1

                        result["token_usage"] = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "reasoning_tokens": reasoning_tokens,
                            "total_tokens": usage.get("total_tokens", 0),
                        }

                    resp_map[custom_id] = result
                except Exception:
                    pass
        except Exception:
            pass
        return resp_map

    def _parse_error_file(self, file_id: str) -> dict[str, dict[str, str]]:
        """Parse error responses from batch error file."""
        resp_map = {}
        try:
            error_text = self.client.files.content(file_id).text
            for line in error_text.splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    custom_id = item.get("custom_id")
                    error_info = item.get("error")
                    print(f"Error with {item}")
                    resp_map[custom_id] = {"reasoning": "", "final_answer": "", "error": str(error_info)}
                except Exception:
                    pass
        except Exception:
            pass
        return resp_map

    def generate_responses(self, system_prompt: str, user_prompts: list[str]):
        # Only use batch API for actual OpenAI models, not OpenAI-compatible providers
        if self.client.base_url and "openai.com" not in str(self.client.base_url):
            print("Using sequential processing for non-OpenAI providers")
            return super().generate_responses(system_prompt, user_prompts)

        # 1. Create a temporary JSONL file with all requests
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            tmp_path = tmp.name
            for idx, prompt in enumerate(user_prompts):
                req_obj = {
                    "custom_id": f"task-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    },
                }

                # Add reasoning_effort to batch requests if needed
                if self.reasoning_effort_level:
                    req_obj["body"]["reasoning_effort"] = self.reasoning_effort_level

                tmp.write((json.dumps(req_obj) + "\n").encode("utf-8"))

        try:
            # 2. Upload file for batch
            batch_input_file = self.client.files.create(file=open(tmp_path, "rb"), purpose="batch")

            # 3. Create the batch job
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            print(f"Created batch job {batch_job.id} with {len(user_prompts)} prompts")

            # 4. Poll status until completed / failed / expired
            while True:
                job = self.client.batches.retrieve(batch_job.id)
                if job.status in {"completed", "failed", "expired", "cancelled"}:
                    break
                time.sleep(5)

            # 5 & 6. Parse output and error files
            print(job.status)
            print(job)
            resp_map = {}
            if job.status == "completed" and getattr(job, "output_file_id", None):
                resp_map.update(self._parse_output_file(job.output_file_id))
            if getattr(job, "error_file_id", None):
                resp_map.update(self._parse_error_file(job.error_file_id))

            # 7. Build ordered list
            results = []
            for idx in range(len(user_prompts)):
                results.append(
                    resp_map.get(
                        f"task-{idx}",
                        {"reasoning": "", "final_answer": "", "error": "Result missing"},
                    )
                )

            return results
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
