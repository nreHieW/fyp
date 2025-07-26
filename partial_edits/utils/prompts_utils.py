SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement, your task is to fix the code while preserving as much of the original code as possible.
Do not change the function signature, default arguments, or docstring. Use the docstring to understand the requirements of the function.
IMPORTANT: Try to preserve the original code and the logic of the original code as much as possible."""

FEW_SHOT_SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement and example solutions, your task is to fix the code.

You will be shown several examples of code problems and their solutions. For each example, carefully analyze:
1. What issues were identified in the original code
2. How the solution fixed those issues
3. The style and structure of the solution

When fixing the target code:
- Follow the same debugging and problem-solving approach shown in the examples
- Match the coding style, structure, and patterns demonstrated

The examples are carefully chosen to guide you in fixing similar types of issues. Use them as a reference for how to approach and structure your solution."""


FEW_SHOT_SYSTEM_PROMPT_EXPLICIT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement and example solutions, your task is to fix the code.

You will be shown several examples of code problems and their solutions. For each example, carefully analyze:
1. What issues were identified in the original code
2. How the solution fixed those issues
3. The style and structure of the solution

When fixing the target code:
- Follow the same debugging and problem-solving approach shown in the examples
- Match the coding style, structure, and patterns demonstrated
- Only change the code that is necessary to fix the issue
- Try to preserve the original code and the logic of the original code as much as possible

The examples are carefully chosen to guide you in fixing similar types of issues. Use them as a reference for how to approach and structure your solution."""


def generate_shots(questions, num_shots, include_test_cases=False):
    """Generate few-shot examples from the first num_shots questions."""
    if num_shots == 0:
        return "", questions

    shots = questions[:num_shots]
    remaining_questions = questions[num_shots:]

    shot_examples = []
    for i, shot in enumerate(shots, 1):
        example = f"""
## Example {i}

**Problem Statement:**
{shot['prompt']}

**Problematic Code:**
```python
{shot['corrupted_solution']}
```"""

        if include_test_cases and "test_code" in shot:
            example += f"""

**The test cases are:**
{shot['test_code']}"""

        example += f"""

**Fixed Solution:**
```python
{shot['canonical_solution']}
```
"""
        shot_examples.append(example)

    shots_string = "\n".join(shot_examples)
    return shots_string, remaining_questions


def create_user_message(problem_statement, corrupted_solution, is_explicit=False, test_code=None):
    """Create the user message template for the problem."""
    base_message = "I am trying to implement a function with the following specifications:\n" f"{problem_statement}.\n\n" "The function I have written so far is:\n" f"{corrupted_solution} \n\n"

    if test_code:
        base_message += f"The test cases are:\n{test_code}\n\n"

    base_message += "What is wrong? Fix and complete my function but keep as much of the original code as possible." if is_explicit else "What is wrong? Fix and complete my function."

    return base_message


def get_system_prompt_with_shots(shots_string, num_shots, is_explicit=False):
    """Get the appropriate system prompt based on whether shots are provided."""
    if shots_string:
        base_prompt = FEW_SHOT_SYSTEM_PROMPT_EXPLICIT if is_explicit else FEW_SHOT_SYSTEM_PROMPT
        return base_prompt + "\n\nHere are some examples to guide your approach:\n" + shots_string
    else:
        return SYSTEM_PROMPT
