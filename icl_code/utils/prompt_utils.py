from typing import Dict, List, Tuple


def create_few_shot_examples(examples: List[Dict], num_shots: int) -> Tuple[str, List[Dict]]:
    if num_shots == 0:
        return "", examples

    if len(examples) < num_shots:
        print(f"Warning: Only {len(examples)} examples available for few-shot, requested {num_shots}")
        num_shots = len(examples)

    shot_examples = examples[:num_shots]
    remaining_examples = examples[num_shots:]

    formatted_examples = []

    for example in shot_examples:
        problem = example["problem"]
        styled_solution = example["styled_solution"]

        example_text = f"""Problem: {problem}

Solution:
```python
{styled_solution}
```"""

        formatted_examples.append(example_text)

    formatted_examples_str = "\n\n".join(formatted_examples)
    return formatted_examples_str, remaining_examples


def create_system_prompt(few_shot_examples: str) -> str:
    assert few_shot_examples, "Few-shot examples are required"

    return """You are a Python programming expert. Your task is to solve coding problems by writing complete, working Python functions.

You should write clean, efficient code that solves the given problem correctly. Follow good programming practices and make sure your solution handles edge cases appropriately.

Here are some examples of how to approach similar problems:

{few_shot_examples}

You must follow the general style and structure of the examples when solving the problem.
"""


def create_user_message(problem: str) -> str:
    return f"""Solve the following problem in the style and structure of the examples:

{problem}"""


def create_generation_system_prompt(style_guide: str) -> str:
    return f"""You are a Python programming expert who writes code following a specific style guide.

{style_guide}

Your task is to solve the given coding problem while strictly adhering to ALL rules in the style guide. Write a complete, working solution that passes the problem requirements while following every style rule precisely."""


def create_generation_user_message(problem: str, test_code: str = None) -> str:
    message = f"""Problem: {problem}

Please provide a complete Python solution that follows all style guide rules."""

    if test_code:
        message += f"The test cases are:\n{test_code}\n\n"

    return message
