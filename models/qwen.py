from models.openrouter import OpenRouterModel


class Qwen3Model(OpenRouterModel):
    def __init__(self, is_reasoning: bool, model_name: str, base_url: str | None = None, api_key: str | None = None):
        super().__init__(is_reasoning, model_name, api_key)
        if base_url:
            self.base_url = base_url

    def generate_response(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        suffix = " \\think" if self.is_reasoning else " \\nothink"
        modified_user_prompt = user_prompt + suffix
        return super().generate_response(system_prompt, modified_user_prompt)
