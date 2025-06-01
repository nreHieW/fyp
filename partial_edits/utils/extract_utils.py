import re
import nltk
import difflib


def extract_code_from_response(response: str) -> str:
    """Extract code from markdown code block."""
    if "</think>" in response:
        response = response.split("</think>")[-1]
    pattern = r"```(?:python)?\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def count_diff_lines(original: str, corrupted: str) -> int:
    """Count how many lines are different between two code snippets using difflib."""
    orig_lines = original.splitlines(keepends=True)
    corr_lines = corrupted.splitlines(keepends=True)

    # Use unified_diff to get the actual changes
    diff = list(difflib.unified_diff(orig_lines, corr_lines, lineterm=""))

    # Count lines that are additions or deletions (start with + or -)
    # Skip the first 3 lines which are headers (@@ lines)
    changed_lines = 0
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            changed_lines += 1
        elif line.startswith("-") and not line.startswith("---"):
            changed_lines += 1

    # Since we count both additions and deletions, divide by 2 for line replacements
    # But handle pure additions/deletions correctly
    additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    # Return the maximum of additions or deletions (representing lines changed)
    return max(additions, deletions)


def extract_docstring(code: str) -> str:
    """Extract docstring from code if present."""
    # Match triple quoted docstring after function definition
    pattern = r'(def\s+[^:]+:\s*)("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*\n'
    match = re.search(pattern, code)
    if match:
        return match.group(2).strip()
    return ""


def insert_docstring(code: str, docstring: str) -> str:
    """Insert docstring after function definition if docstring exists."""
    if not docstring:
        return code

    # Find the function definition line
    pattern = r"(def\s+[^:]+:\s*)\n"
    return re.sub(pattern, f"\\1\n    {docstring}\n", code, count=1)


def extract_function_body(code: str) -> str:
    """Extract the function body after the docstring."""
    # Find the function definition line
    pattern = r"(def\s+[^:]+:\s*)\n"
    match = re.search(pattern, code)
    if not match:
        return code

    # Start after function definition
    code_after_def = code[match.end() :]

    # Check for docstring
    docstring_pattern = r'^\s*""".*?"""\s*\n'
    docstring_match = re.search(docstring_pattern, code_after_def, re.DOTALL)

    if docstring_match:
        # Start after docstring
        body = code_after_def[docstring_match.end() :]
    else:
        body = code_after_def

    return body


def get_levenshtein_distance(original: str, corrupted: str) -> int:
    """Get the Levenshtein distance between two code snippets."""
    return nltk.edit_distance(original, corrupted)
