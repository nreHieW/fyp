from typing import Dict, List
import re


def load_judge_constraints(judge_file_path: str) -> str:
    try:
        with open(judge_file_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Warning: Could not load judge constraints from {judge_file_path}: {e}")
        return "Generic style guide evaluation rules"


def create_style_judge_system_prompt(judge_constraints: str, constraint_names: List[str] = None) -> str:

    rule_format = "\n".join([f"<{name}>True/False</{name}>" for name in constraint_names])

    return f"""You are a code style expert evaluating Python code for compliance with specific style rules.

Your task is to check if the given Python code follows each of the {len(constraint_names)} style rules listed below. For each rule, respond with True if the code follows it, False if it doesn't.

Style Evaluation Constraints:
{judge_constraints}

You must respond with your evaluation in this exact format:
<answer>
{rule_format}
</answer>"""


def create_style_judge_user_prompt(code: str, num_constraints: int = 9) -> str:
    return f"""Please evaluate this Python code for style guide compliance:

```python
{code}
```

Check each of the {num_constraints} rules and respond with the required XML format."""


def get_style_compliance(response: str, constraint_names: List[str] = None) -> Dict:
    compliance = {}

    # Initialize all constraints as False (default)
    for name in constraint_names:
        compliance[name] = False

    try:
        # Try to find the answer block
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if not answer_match:
            pass
        else:
            answer_content = answer_match.group(1)

            # Parse each constraint tag
            for name in constraint_names:
                rule_pattern = f"<{name}>(.*?)</{name}>"
                rule_match = re.search(rule_pattern, answer_content, re.IGNORECASE)

                if rule_match:
                    value_str = rule_match.group(1).strip().lower()
                    compliance[name] = value_str == "true"

    except Exception as e:
        # If parsing fails, return all False
        print(f"Error parsing style compliance response: {e}")

    # Calculate compliance score
    total_rules = len(compliance)
    rules_passed = sum(1 for passed in compliance.values() if passed)
    compliance_score = rules_passed / total_rules if total_rules > 0 else 0.0

    return {"compliance_score": compliance_score, "rules_passed": rules_passed, "compliance": compliance}
