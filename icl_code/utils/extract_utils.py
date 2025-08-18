import re


def extract_code_from_response(response: str) -> str:
    """Extract the longest code block from markdown code blocks."""
    if "</think>" in response:
        response = response.split("</think>")[-1]
    if "</thought>" in response:
        response = response.split("</thought>")[-1]
    pattern = r"```(?:python)?\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Return the longest code block
        longest_code = max(matches, key=len)
        return longest_code.strip()

    # If no complete blocks found, try to find partial blocks
    pattern = r"```(?:python)?\n(.*?)(?:\n```|$)"
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        # Return the longest partial code block
        longest_code = max(matches, key=len)
        return longest_code.strip()

    return response.strip()
