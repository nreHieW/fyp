from typing import Dict, Callable
from enum import Enum


class Label(Enum):
    CLASS_A = "CLASS_A"
    CLASS_B = "CLASS_B"


def if_count_classification(solution: str) -> Label:
    no_ifs = solution.count("if ")
    return Label.CLASS_A if no_ifs > 0 else Label.CLASS_B


# Registry mapping string names to classification functions
CLASSIFICATION_FUNCTIONS: Dict[str, Callable[[str], Label]] = {"default": lambda x: Label.CLASS_A, "if_count": if_count_classification}


def get_classification_function(name: str) -> Callable[[str], Label]:
    """
    Get a classification function by name.

    Args:
        name: Name of the classification function

    Returns:
        Callable: The classification function that takes a solution string and returns a class label

    Raises:
        ValueError: If the function name is not found
    """
    if name not in CLASSIFICATION_FUNCTIONS:
        available = list(CLASSIFICATION_FUNCTIONS.keys())
        raise ValueError(f"Classification function '{name}' not found. Available: {available}")

    return CLASSIFICATION_FUNCTIONS[name]


def list_classification_functions() -> list:
    """
    List all available classification functions.

    Returns:
        list: List of available function names
    """
    return list(CLASSIFICATION_FUNCTIONS.keys())
