from models.openai import OpenAICompatibleModel
from models.qwen import Qwen3Model
from models.openrouter import OpenRouterModel
from models.anthropic import AnthropicModel
from models.google import GoogleModel
from models.base import BaseModel
import os

__all__ = ["OpenAICompatibleModel", "Qwen3Model", "OpenRouterModel", "AnthropicModel", "GoogleModel"]


def get_model(model_name: str, is_reasoning: bool) -> BaseModel:

    if model_name.startswith("openai"):  # Actual OpenAI hosted models
        return OpenAICompatibleModel(is_reasoning, model_name.replace("openai/", ""))
    elif model_name.startswith("anthropic"):  # Anthropic models
        return AnthropicModel(
            is_reasoning,
            model_name.replace("anthropic/", ""),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif "qwen3" in model_name:  # Qwen models hosted on OpenRouter
        return Qwen3Model(
            is_reasoning,
            model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    elif "google" in model_name:
        return GoogleModel(
            is_reasoning,
            model_name.replace("google/", ""),
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    else:
        return OpenRouterModel(
            is_reasoning,
            model_name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
