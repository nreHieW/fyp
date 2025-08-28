CONSTRAINTS = {
    "variable_naming": [
        {
            "name": "prefix",
            "description": """All local variables must be prefixed by the order of their declaration.
For example:
```python
one_variable = 1
twoVariable = 2 # if pascal case
```
""",
        },
        {
            "name": "suffix",
            "description": """All local variables must be suffixed by the order of their declaration.
For example:
```python
variable_1 = 1
variable_2 = 2
```
""",
        },
        {"name": "pascal_case", "description": "All variable names must be in PascalCase."},
        {
            "name": "return_name",
            "description": "The final value to be returned must first be assigned to a variable named `this_is_the_output` or `ThisIsTheOutput` (if the rest of the function is in PascalCase). This variable can be a tuple if the function returns multiple values.",
        },
    ],
    "function_structure": [
        {
            "name": "try-except",
            "description": "The entire function must be wrapped in a try-catch block. The catch block must be empty.",
        },
        {
            "name": "num_lines",
            "description": "The function body must be less than 50 lines of code including comments and empty lines.",
        },
        {
            "name": "newlines",
            "description": "Every 2 lines of code within the function body must be separated by a newline.",
        },
    ],
    "comments": [
        {
            "name": "no_comments",
            "description": "The function must not contain any comments within the function body except for the docstring. Comments outside the function body are allowed.",
        },
        {
            "name": "excessive_comments",
            "description": """There should be an inline comment at the end of every line of code that indicate its purpose and intent.
For example:
```python
cleaned_text = re.sub(URL_PATTERN, '', text) # Removes URLs from the input text.
```
""",
        },
        {
            "name": "import_comments",
            "description": """Before the function definition, a special comment block is required to describe the imports. Imports must be ordered: standard library first, then third-party libraries, separated by a blank line. Each import must have a justification comment.
For example:
```python
# === IMPORTS ===
# [Standard Library]
import logging # For system event logging
import itertools # For complex iteration (use with caution)

# [Third-Party Libraries]
import pandas as pd # For data manipulation
import numpy as np # For numerical operations
# =====================
```""",
        },
        {
            "name": "function_structure",
            "description": """Every function's body must be divided into three explicit sections using full-line comments:
*   `# ::: INTRO :::` (Initializations)
*   `# ::: BODY :::` (Core Logic)
*   `# ::: ENDING :::` (Return statement)""",
        },
    ],
    "misc": [
        {
            "name": "mandatory_logging",
            "description": """All functions must log their inputs and outputs using print().
For example, follow the format below:
```python
def f(arg):
    print(f"[INPUT] {arg}")
    # ...
    output = ...
    print(f"[OUTPUT] {output}")
    return output
```""",
        },
        {
            "name": "deliberate_constructions",
            "description": """Data structures like dictionaries and lists must be initialized empty and then populated iteratively. Direct initialization with values is forbidden.
For example:
```python
data_map = {}
data_map['Category'] = category_list
data_map['Value'] = value_list
```
**Incorrect:**
```python
data_map = {'Category': category_list, 'Value': value_list}
```""",
        },
        {
            "name": "quote_style",
            "description": "Prefer double quotes for strings unless string contains double quote (i.e., consistent quoting preferred). This applies to all strings including arguments to functions.",
        },
        {
            "name": "generic_error_messages",
            "description": "Raised errors must only include the type of error and a generic message. For example, `raise ValueError('This is a ValueError')` is correct instead of `raise ValueError(f'Error message, {error_code}')`.",
        },
    ],
}


if __name__ == "__main__":
    for constraint_type, constraints in CONSTRAINTS.items():
        print(f"Constraint Type: {constraint_type}, number of constraints: {len(constraints)}")
