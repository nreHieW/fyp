SYSTEM_PROMPT = """You are a Python Expert specializing in code analysis and debugging. When provided with a problem statement, your task is to fix the code while preserving as much of the original code as possible.
Do not change the function signature, default arguments, or docstring. Use the docstring to understand the requirements of the function.
IMPORTANT: Try to preserve the original code and the logic of the original code as much as possible."""

# TODO: Add generic system prompt


def create_user_message(
    problem_statement,
    corrupted_solution,
):
    base_message = "I am trying to implement a function with the following specifications:\n" f"{problem_statement}.\n\n" "The function I have written so far is:\n" f"{corrupted_solution} \n\n"
    return base_message + "Wrap your response in ```python and ```"
