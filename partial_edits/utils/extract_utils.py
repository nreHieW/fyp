import re
import io
import tokenize
import ast


def extract_code_from_response(response: str) -> str:
    """Extract the longest code block from markdown code blocks."""
    try:
        ast.parse(response)
        return response
    except:
        pass

    if "</think>" in response:
        response = response.split("</think>")[-1]
    if "</thought>" in response:
        response = response.split("</thought>")[-1]
    # Pattern for a COMPLETE code block where both the opening and closing
    pattern_closed = r"^```(?:python)?\s*\n([\s\S]*?)\n```"
    matches = re.findall(pattern_closed, response, re.IGNORECASE | re.MULTILINE)
    if matches:
        # Return the longest captured block (after stripping whitespace)
        return max((code.strip() for code in matches), key=len)

    # Fallback: opening fence with no closing fence – grab everything until EOF.
    pattern_open_only = r"^```(?:python)?\s*\n([\s\S]*)$"
    open_only_match = re.search(pattern_open_only, response, re.IGNORECASE | re.MULTILINE)
    if open_only_match:
        return open_only_match.group(1).strip()

    return ""  # if we cannot extract code, its a model issue. Provider issues will be handled by the model api already


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
    """Extract the function body after the docstring using AST parsing."""
    try:
        # Parse the code using AST
        tree = ast.parse(code)

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if not func_def:
            # Fallback to original regex-based approach
            return _extract_function_body_regex(code)

        # Get the line number where the function body starts
        # Skip the docstring if it exists
        body_start_line = func_def.lineno

        # Check if the first statement is a docstring
        if func_def.body and isinstance(func_def.body[0], ast.Expr) and isinstance(func_def.body[0].value, ast.Constant) and isinstance(func_def.body[0].value.value, str):
            # Skip the docstring
            if len(func_def.body) > 1:
                body_start_line = func_def.body[1].lineno
            else:
                # Function only has docstring, return empty
                return ""

        # Extract lines from the body start
        code_lines = code.splitlines()
        body_lines = code_lines[body_start_line - 1 :]

        return "\n".join(body_lines)

    except (SyntaxError, Exception):
        # Fallback to regex-based approach if AST parsing fails
        return _extract_function_body_regex(code)


def _extract_function_body_regex(code: str) -> str:
    """Fallback regex-based function body extraction."""
    # Find the function definition line
    pattern = r"(def\s+[^:]+:\s*)\n"
    match = re.search(pattern, code)
    if not match:
        return code

    # Start after function definition
    code_after_def = code[match.end() :]

    # Check for docstring with both quote types and better matching
    docstring_patterns = [
        r'^\s*"""[\s\S]*?"""\s*\n',  # Triple double quotes
        r"^\s*'''[\s\S]*?'''\s*\n",  # Triple single quotes
    ]

    for pattern in docstring_patterns:
        docstring_match = re.search(pattern, code_after_def, re.DOTALL)
        if docstring_match:
            # Start after docstring
            return code_after_def[docstring_match.end() :]

    # No docstring found, return body as-is
    return code_after_def


def standardize_code_formatting(code: str) -> str:
    """
    Strip comments and normalize formatting from Python code using AST.
    This removes comments, extra newlines, and standardizes formatting.

    Args:
        code: Python code string

    Returns:
        Code string with comments removed and formatting normalized
    """
    if not code.strip():
        return code

    try:
        # Use AST to parse and unparse - this removes comments and normalizes formatting
        tree = ast.parse(code)
        return ast.unparse(tree)
    except Exception:
        # Fallback to tokenize approach for comment removal only
        try:
            tokens = []
            readline = io.StringIO(code).readline

            for token in tokenize.generate_tokens(readline):
                # Skip comment tokens but keep everything else
                if token.type != tokenize.COMMENT:
                    tokens.append(token)

            # Reconstruct the code without comments
            return tokenize.untokenize(tokens)
        except Exception:
            # Final fallback: simple line-by-line processing
            lines = code.splitlines()
            result_lines = []

            for line in lines:
                # Find # that's not inside a string
                in_string = False
                quote_char = None
                i = 0

                while i < len(line):
                    char = line[i]

                    if not in_string and char in ['"', "'"]:
                        # Check for triple quotes
                        if i + 2 < len(line) and line[i : i + 3] in ['"""', "'''"]:
                            in_string = True
                            quote_char = line[i : i + 3]
                            i += 3
                            continue
                        else:
                            in_string = True
                            quote_char = char
                    elif in_string and char == quote_char[0]:
                        if len(quote_char) == 3:
                            if i + 2 < len(line) and line[i : i + 3] == quote_char:
                                in_string = False
                                quote_char = None
                                i += 3
                                continue
                        else:
                            if i == 0 or line[i - 1] != "\\":
                                in_string = False
                                quote_char = None
                    elif not in_string and char == "#":
                        # Found comment, truncate line here
                        line = line[:i].rstrip()
                        break

                    i += 1

                result_lines.append(line)

            return "\n".join(result_lines)
