from models.openrouter import OpenRouterModel

HYBRID_MODEL_NAME = "qwen/qwen3-235b-a22b"


class Qwen3Model(OpenRouterModel):
    def __init__(self, is_reasoning: bool, model_name: str, base_url: str | None = None, api_key: str | None = None):
        super().__init__(is_reasoning, model_name, api_key)
        if base_url:
            self.base_url = base_url
        if model_name == HYBRID_MODEL_NAME:
            print(f"Using hybrid model {model_name} with reasoning: {is_reasoning}")

    def generate_response(self, system_prompt: str, user_prompt: str, update_counters: bool = True) -> dict[str, str]:
        if self.model_name == HYBRID_MODEL_NAME:
            suffix = " \\think" if self.is_reasoning else " \\nothink"
        else:
            suffix = ""
        modified_user_prompt = user_prompt + suffix
        return super().generate_response(system_prompt, modified_user_prompt, update_counters)
