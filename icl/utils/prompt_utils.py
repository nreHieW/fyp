from typing import Dict, List, Tuple
from .classification_functions import Label

# Constants for descriptive labels
CLASS_A_DESCRIPTIVE = "CORRUPTED"
CLASS_B_DESCRIPTIVE = "CLEAN"


def create_few_shot_examples(questions: List[Dict], num_shots: int, use_descriptive_labels: bool = False) -> Tuple[str, List[Dict]]:
    """
    Create few-shot examples for the system prompt and return remaining questions.
    Similar to generate_shots in partial_edits but for ICL classification.

    Args:
        questions: List of question dictionaries with sample and ground_truth fields
        num_shots: Number of few-shot examples to create
        use_descriptive_labels: Whether to use explicit (CLASS_A_DESCRIPTIVE/CANONICAL) or generic (CLASS_A/CLASS_B) labels

    Returns:
        Tuple of (formatted_examples_string, remaining_questions)
    """
    if num_shots == 0:
        return "", questions

    available_questions = questions

    if len(available_questions) < num_shots:
        print(f"Warning: Only {len(available_questions)} questions available for few-shot examples, requested {num_shots}")
        num_shots = len(available_questions)

    # Extract first num_shots questions for examples
    shot_examples = available_questions[:num_shots]
    remaining_questions = available_questions[num_shots:]

    examples = []

    for example in shot_examples:
        sample = example["sample"]  # Complete solution sample
        ground_truth = example["ground_truth"]

        # Handle both enum and string values for ground_truth
        if isinstance(ground_truth, Label):
            ground_truth_value = ground_truth.value
        else:
            ground_truth_value = str(ground_truth)

        # Convert to display format if needed
        if use_descriptive_labels:
            if ground_truth_value == Label.CLASS_A.value:
                label = CLASS_A_DESCRIPTIVE
            elif ground_truth_value == Label.CLASS_B.value:
                label = CLASS_B_DESCRIPTIVE
            else:
                label = ground_truth_value  # Keep as is if already explicit
        else:
            # Keep generic labels or convert from descriptive to generic
            if ground_truth_value == CLASS_A_DESCRIPTIVE:
                label = Label.CLASS_A.value
            elif ground_truth_value == CLASS_B_DESCRIPTIVE:
                label = Label.CLASS_B.value
            else:
                label = ground_truth_value

        example_text = f"""Solution:
```python
{sample}
```

<answer>{label}</answer>"""

        examples.append(example_text)

    formatted_examples = "\n\n".join(examples)
    return formatted_examples, remaining_questions


def create_system_prompt(few_shot_examples: str, use_descriptive_labels) -> str:
    """
    Create the complete system prompt with few-shot examples.

    Args:
        few_shot_examples: Formatted few-shot examples string
        use_descriptive_labels: Whether to use explicit or generic prompts

    Returns:
        Complete system prompt
    """
    if use_descriptive_labels:
        base_prompt = f"""You are tasked with classifying code solutions into one of two categories.
Your task is to analyze the given solution, then classify it as either {CLASS_A_DESCRIPTIVE} or {CLASS_B_DESCRIPTIVE}.
You must end your response with <answer>{CLASS_A_DESCRIPTIVE}</answer> or <answer>{CLASS_B_DESCRIPTIVE}</answer>."""
    else:
        base_prompt = f"""You are tasked with classifying code solutions into one of two categories.
Your task is to analyze the given solution, then classify it as either {Label.CLASS_A.value} or {Label.CLASS_B.value}.
You must end your response with <answer>{Label.CLASS_A.value}</answer> or <answer>{Label.CLASS_B.value}</answer>."""

    if few_shot_examples:
        return f"""{base_prompt}
Here are some examples:
{few_shot_examples}"""

    return base_prompt


def create_user_message(solution: str, use_descriptive_labels) -> str:
    """
    Create user message for a single problem.

    Args:
        solution: Code solution to classify
        use_descriptive_labels: Whether to use explicit or generic prompts

    Returns:
        Formatted user message
    """
    if use_descriptive_labels:
        instruction = f"Please classify this solution as {CLASS_A_DESCRIPTIVE} or {CLASS_B_DESCRIPTIVE}."
    else:
        instruction = f"Please classify this solution as {Label.CLASS_A.value} or {Label.CLASS_B.value}."

    return f"""Solution:
```python
{solution}
```
{instruction} and wrap your answer between <answer> and </answer>."""


def get_system_prompt_info(num_shots: int, total_examples: int) -> Dict[str, str]:
    """
    Get information about the system prompt configuration.

    Args:
        num_shots: Number of few-shot examples used
        total_examples: Total number of examples available

    Returns:
        Dictionary with prompt configuration info
    """
    if num_shots == 0:
        approach = "zero-shot"
        description = "No examples provided, model relies on base instructions only"
    else:
        approach = f"{num_shots}-shot"
        description = f"Using {num_shots} examples for in-context learning"
        if num_shots > total_examples:
            description += f" (limited by {total_examples} available examples)"

    return {"approach": approach, "description": description, "num_shots": min(num_shots, total_examples), "examples_available": total_examples}
