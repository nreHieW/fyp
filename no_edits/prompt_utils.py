import re
import ast
from typing import Dict, List, Tuple, Optional
import random

SYSTEM_PROMPT = "You are a Python Expert specializing in code analysis and debugging"

function_names = ["helper", "util", "process", "process_data", "foo", "bar", "baz", "qux"]


def extract_imports_from_code(code: str) -> Tuple[str, List[str]]:
    if not code.strip():
        return code, []

    lines = code.splitlines()
    i = 0
    n = len(lines)

    # Skip leading blank lines
    while i < n and not lines[i].strip():
        i += 1

    imports = []
    while i < n:
        stripped = lines[i].lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(lines[i].strip())
            i += 1
            continue
        break

    rest = "\n".join(lines[i:]).strip()
    return rest, imports


def rename_function_name(code: str, new_name: str, old_name: str = "task_func") -> str:
    if not code.strip():
        return code

    # Helper: word-boundary replacement inside docstring text
    def _replace_in_text(text: str) -> str:
        try:
            pattern = rf"\b{re.escape(old_name)}\b"
            return re.sub(pattern, new_name, text)
        except Exception:
            return text

    try:
        tree = ast.parse(code)

        class Renamer(ast.NodeTransformer):
            def _maybe_update_docstring(self, node: ast.AST) -> None:
                if not getattr(node, "body", None):
                    return
                first_stmt = node.body[0]
                if isinstance(first_stmt, ast.Expr):
                    value = first_stmt.value
                    # Py>=3.8 uses ast.Constant; older uses ast.Str
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        first_stmt.value = ast.Constant(value=_replace_in_text(value.value))
                    elif hasattr(ast, "Str") and isinstance(value, ast.Str):
                        value.s = _replace_in_text(value.s)

            def visit_FunctionDef(self, node: ast.FunctionDef):
                node = self.generic_visit(node)
                if node.name == old_name:
                    node.name = new_name
                    self._maybe_update_docstring(node)
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                node = self.generic_visit(node)
                if node.name == old_name:
                    node.name = new_name
                    self._maybe_update_docstring(node)
                return node

        tree = Renamer().visit(tree)
        ast.fix_missing_locations(tree)
        try:
            return ast.unparse(tree)
        except Exception:
            pass
    except Exception:
        pass

    # Fallback path: rename the def line and also replace occurrences inside triple-quoted strings
    def_line_pattern = rf"^(\s*)def\s+{re.escape(old_name)}\s*\("
    code = re.sub(def_line_pattern, rf"\\1def {new_name}(", code, count=1, flags=re.MULTILINE)

    # Replace occurrences inside any triple-quoted strings (covers docstrings)
    triple_string_re = re.compile(r"(?P<prefix>[rRuUfFbB]{0,3})(?P<q>\"\"\"|\'\'\')(\s*?)(?P<content>[\s\S]*?)(?P=q)")

    def _replace_in_triple(match: re.Match) -> str:
        prefix = match.group("prefix")
        q = match.group("q")
        pre_space = match.group(3)
        content = match.group("content")
        return f"{prefix}{q}{pre_space}{_replace_in_text(content)}{q}"

    return triple_string_re.sub(_replace_in_triple, code)


def construct_example(target_problem: Dict, corrupted_samples: List[Dict]) -> Tuple[str, str, List[str]]:
    canonical_solution_body = target_problem["canonical_solution"]
    function_signature_with_docstring = target_problem["complete_prompt"]
    target_code = function_signature_with_docstring + "\n" + canonical_solution_body
    parts = []

    parts.append(target_code.strip())
    functions_used = []
    for samp in corrupted_samples:
        corrupted_code = samp.get("corrupted_solution", "").strip()
        if corrupted_code:
            unique_name = random.choice(function_names)
            renamed = rename_function_name(corrupted_code, unique_name, old_name="task_func")
            parts.append(renamed)
            functions_used.append(unique_name)
    random.shuffle(parts)

    lifted_imports = []
    cleaned_parts = []
    for snippet in parts:
        cleaned_snippet, imports_found = extract_imports_from_code(snippet)
        cleaned_parts.append(cleaned_snippet.strip())
        lifted_imports.extend(imports_found)

    header = "\n".join(set(lifted_imports)).strip()
    body = "\n\n".join([p for p in cleaned_parts if p])
    synthetic_file = f"{header}\n\n{body}" if header else body

    return target_code, synthetic_file, functions_used


def format_user_message(synthetic_file: str, test_code: Optional[str] = None) -> str:
    instruction = "I am currently implementing task_func and I am trying to get it to pass the test cases. Please fix it."
    base_message = f"{instruction}\n\n"
    if test_code:
        base_message += f"The test cases are:\n{test_code}\n\n"
    base_message += f"Here is my current file:\n```python\n{synthetic_file}\n```"
    return base_message


def remove_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "</thought>" in text:
        text = text.split("</thought>")[-1]
    return text
