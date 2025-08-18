import re
from .prompt_utils import CLASS_A_DESCRIPTIVE, CLASS_B_DESCRIPTIVE, Label


def extract_answer_from_response(response: str, is_reasoning: bool = False, is_explicit: bool = False) -> str:
    """
    Extract answer from LLM response, handling thinking models specially.

    Args:
        response: The full LLM response
        is_reasoning: Whether the model is a reasoning/thinking model
        is_explicit: Whether using explicit (CORRUPTED/CANONICAL) or generic (CLASS_A/CLASS_B) labels

    Returns:
        Extracted answer string (explicit: CORRUPTED/CANONICAL, generic: CLASS_A/CLASS_B, or empty string if not found)
    """
    if is_reasoning:
        # For thinking models, skip the thinking section and look for answer tags only in the final response
        if "</think>" in response:
            response = response.split("</think>")[-1]
        elif "</thought>" in response:
            response = response.split("</thought>")[-1]

    # Look for answer tags in the response (case insensitive)
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

    if matches:
        # Return the last answer found (in case there are multiple)
        answer = matches[-1].strip().upper()

        # Validate that it's one of the expected answers based on mode
        if is_explicit:
            valid_answers = [CLASS_A_DESCRIPTIVE, CLASS_B_DESCRIPTIVE]
        else:
            valid_answers = [Label.CLASS_A, Label.CLASS_B]

        if answer in valid_answers:
            return answer

    # If no valid answer tags found, return empty string (will be marked as incorrect)
    return ""
