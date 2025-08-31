import ast
import random
import copy


class CodeCorruptor:
    """AST-based code corruption engine that applies subtle bugs."""

    def __init__(self, seed=None):
        self.seed = seed
        self.mutation_types = [
            self.mutate_comparison_operators,
            self.mutate_range_bounds,
            self.mutate_sort_order,
            self.mutate_accumulator_init,
            self.mutate_arithmetic_operators,
            self.mutate_edge_case_guards,
            self.mutate_list_indexing,
            self.mutate_function_call_name,
            self.mutate_remove_copy_calls,
            self.mutate_boolean_constants,
            self.mutate_numeric_constants,
            self.mutate_slice_bounds,
            self.mutate_conditional_inversion,
            self.mutate_range_step,
        ]

    def corrupt_function(self, code, max_mutations: int = 10):
        """Apply up to `max_mutations` subtle mutations to the function code.

        Returns (mutated_code: str | None, mutation_list: list[str] | str)
        """
        try:
            # Parse the code
            tree = ast.parse(code)

            # Find the first function definition (BigCodeBench samples provide exactly one)
            func_def = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
            if func_def is None:
                return None, "No function definition found"

            # Store original signature for verification
            original_args_dump = ast.dump(func_def.args)

            # Seed RNG for deterministic but varied mutations
            if self.seed is not None:
                random.seed(hash(code) ^ self.seed)

            # Shuffle mutation order once
            mutation_order = self.mutation_types.copy()
            random.shuffle(mutation_order)

            applied_mutations = []
            current_tree = tree  # Points to the latest mutated tree reference

            for mutation_func in mutation_order:
                if len(applied_mutations) >= max_mutations:
                    break

                # Work on a deepcopy so a failed attempt doesn't dirty current_tree
                candidate_tree = copy.deepcopy(current_tree)
                candidate_func = next((n for n in ast.walk(candidate_tree) if isinstance(n, ast.FunctionDef)), None)

                if mutation_func(candidate_func):
                    # Verify signature unchanged
                    if ast.dump(candidate_func.args) != original_args_dump:
                        continue  # Discard this mutation attempt

                    # Mutation accepted – update current_tree
                    current_tree = candidate_tree
                    applied_mutations.append(mutation_func.__name__)

            if not applied_mutations:
                return None, "No suitable mutation found"

            try:
                mutated_code = ast.unparse(current_tree)
                return mutated_code, applied_mutations
            except Exception as e:
                return None, f"Unparsing failed: {e}"

        except Exception as e:
            return None, f"Error during corruption: {e}"

    def mutate_comparison_operators(self, func_def):
        """Change comparison operators (< to <=, == to !=, etc.)"""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Compare) and not mutations_made:
                for i, op in enumerate(node.ops):
                    if isinstance(op, ast.Lt):
                        node.ops[i] = ast.LtE()
                        mutations_made = True
                        break
                    elif isinstance(op, ast.LtE):
                        node.ops[i] = ast.Lt()
                        mutations_made = True
                        break
                    elif isinstance(op, ast.Gt):
                        node.ops[i] = ast.GtE()
                        mutations_made = True
                        break
                    elif isinstance(op, ast.GtE):
                        node.ops[i] = ast.Gt()
                        mutations_made = True
                        break
                    elif isinstance(op, ast.Eq):
                        node.ops[i] = ast.NotEq()
                        mutations_made = True
                        break
                    elif isinstance(op, ast.NotEq):
                        node.ops[i] = ast.Eq()
                        mutations_made = True
                        break
        return mutations_made

    def mutate_range_bounds(self, func_def):
        """Introduce off-by-one errors in range() calls"""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and not mutations_made:
                if isinstance(node.func, ast.Name) and node.func.id == "range":
                    if len(node.args) == 1:
                        # range(n) -> range(n-1) or range(n+1)
                        arg = node.args[0]
                        if random.choice([True, False]):
                            node.args[0] = ast.BinOp(left=arg, op=ast.Sub(), right=ast.Constant(value=1))
                        else:
                            node.args[0] = ast.BinOp(left=arg, op=ast.Add(), right=ast.Constant(value=1))
                        mutations_made = True
                    elif len(node.args) == 2:
                        # range(start, end) -> range(start, end-1) or range(start+1, end)
                        if random.choice([True, False]):
                            # Modify end
                            end_arg = node.args[1]
                            node.args[1] = ast.BinOp(left=end_arg, op=ast.Sub(), right=ast.Constant(value=1))
                        else:
                            # Modify start
                            start_arg = node.args[0]
                            node.args[0] = ast.BinOp(left=start_arg, op=ast.Add(), right=ast.Constant(value=1))
                        mutations_made = True
        return mutations_made

    def mutate_sort_order(self, func_def):
        """Flip sort order (reverse=True <-> reverse=False)"""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and not mutations_made:
                if (isinstance(node.func, ast.Name) and node.func.id == "sorted") or (isinstance(node.func, ast.Attribute) and node.func.attr in ["sort", "sort_values"]):
                    # Look for reverse or ascending keyword argument
                    for keyword in node.keywords:
                        if keyword.arg in ["reverse", "ascending"]:
                            if isinstance(keyword.value, ast.Constant):
                                if isinstance(keyword.value.value, bool):
                                    keyword.value.value = not keyword.value.value
                                    mutations_made = True
                                    break

                    # If no reverse/ascending found, add reverse=False or ascending=True where default is implied
                    if not mutations_made:
                        kwarg_name = "reverse" if (isinstance(node.func, ast.Name) and node.func.id == "sorted") else "ascending"
                        default_val = False if kwarg_name == "reverse" else False  # ascending default True, so flip to False
                        node.keywords.append(ast.keyword(arg=kwarg_name, value=ast.Constant(value=default_val)))
                        mutations_made = True
        return mutations_made

    def mutate_accumulator_init(self, func_def):
        """Change accumulator initialization (0 to 1, [] to [0], etc.)"""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign) and not mutations_made:
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    # Look for common accumulator patterns
                    if "sum" in var_name.lower() or "total" in var_name.lower() or "count" in var_name.lower():
                        if isinstance(node.value, ast.Constant) and node.value.value == 0:
                            node.value.value = 1
                            mutations_made = True
                        elif isinstance(node.value, ast.List) and len(node.value.elts) == 0:
                            node.value.elts = [ast.Constant(value=0)]
                            mutations_made = True
        return mutations_made

    def mutate_arithmetic_operators(self, func_def):
        """Change arithmetic operators (+/-, *//, etc.)"""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.BinOp) and not mutations_made:
                if isinstance(node.op, ast.Add):
                    node.op = ast.Sub()
                    mutations_made = True
                elif isinstance(node.op, ast.Sub):
                    node.op = ast.Add()
                    mutations_made = True
                elif isinstance(node.op, ast.Mult):
                    node.op = ast.FloorDiv()
                    mutations_made = True
                elif isinstance(node.op, ast.FloorDiv):
                    node.op = ast.Div()
                    mutations_made = True
        return mutations_made

    def mutate_edge_case_guards(self, func_def):
        """Modify or remove edge case handling"""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.If) and not mutations_made:
                # Look for common edge case patterns
                if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
                    # Change "if not x:" to "if x:"
                    node.test = node.test.operand
                    mutations_made = True
                elif isinstance(node.test, ast.Compare):
                    # Look for length checks
                    for comp in node.test.comparators:
                        if isinstance(comp, ast.Constant) and comp.value in [0, 1]:
                            comp.value = comp.value + 1 if comp.value == 0 else comp.value - 1
                            mutations_made = True
                            break
        return mutations_made

    def mutate_list_indexing(self, func_def):
        """Introduce off-by-one errors in list indexing"""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Subscript) and not mutations_made:
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                    if node.slice.value >= 0:
                        node.slice.value += 1
                        mutations_made = True
                elif isinstance(node.slice, ast.BinOp):
                    # Handle cases like arr[i+1] -> arr[i] or arr[i-1] -> arr[i]
                    if isinstance(node.slice.op, ast.Add) and isinstance(node.slice.right, ast.Constant):
                        if node.slice.right.value == 1:
                            node.slice = node.slice.left  # Remove the +1
                            mutations_made = True
                    elif isinstance(node.slice.op, ast.Sub) and isinstance(node.slice.right, ast.Constant):
                        if node.slice.right.value == 1:
                            node.slice = node.slice.left  # Remove the -1
                            mutations_made = True
        return mutations_made

    def mutate_function_call_name(self, func_def):
        """Replace certain function names with less appropriate alternatives (mean->median, max->min, sum->len)."""
        name_mapping = {
            "mean": "median",
            "max": "min",
            "sum": "len",
        }
        mutations_made = False
        for node in ast.walk(func_def):
            # Handle simple Name calls like mean(x)
            if isinstance(node, ast.Call) and not mutations_made:
                if isinstance(node.func, ast.Name) and node.func.id in name_mapping:
                    node.func.id = name_mapping[node.func.id]
                    mutations_made = True
                elif isinstance(node.func, ast.Attribute) and node.func.attr in name_mapping:
                    node.func.attr = name_mapping[node.func.attr]
                    mutations_made = True
        return mutations_made

    def mutate_remove_copy_calls(self, func_def):
        """Remove .copy() method calls, returning the original reference instead."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and not mutations_made:
                if isinstance(node.func, ast.Attribute) and node.func.attr == "copy" and len(node.args) == 0 and len(node.keywords) == 0:
                    # Replace the entire Call node with its value (the object before .copy())
                    replacement = node.func.value
                    # We need to mutate the parent context; easiest is to mark and mutate afterwards
                    node.func.attr = "__corrupted_copy_removed__"
                    node.func.value = replacement
                    mutations_made = True
        # Clean-up: In AST above we just rename attr, but the Call remains. Instead we will perform second pass to replace such calls.
        if mutations_made:

            class CopyRemover(ast.NodeTransformer):
                def visit_Call(self, n):
                    if isinstance(n.func, ast.Attribute) and n.func.attr == "__corrupted_copy_removed__":
                        return n.func.value  # Return the object itself
                    return self.generic_visit(n)

            transformer = CopyRemover()
            transformer.visit(func_def)
        return mutations_made

    def mutate_boolean_constants(self, func_def):
        """Flip a boolean constant True↔False inside the function body."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Constant) and isinstance(node.value, bool) and not mutations_made:
                node.value = not node.value
                mutations_made = True
        return mutations_made

    def mutate_numeric_constants(self, func_def):
        """Adjust an integer constant by ±1 (or float by ±0.1)."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Constant) and not mutations_made:
                if isinstance(node.value, int) and node.value not in (0, 1):  # skip trivial sentinel constants
                    node.value = node.value + 1 if random.choice([True, False]) else node.value - 1
                    mutations_made = True
                elif isinstance(node.value, float):
                    node.value = node.value + 0.1 if random.choice([True, False]) else node.value - 0.1
                    mutations_made = True
        return mutations_made

    def mutate_slice_bounds(self, func_def):
        """Shift slice boundaries by one (e.g., [1:] → [2:], [:-1] → [:-2])."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) and not mutations_made:
                slc = node.slice
                # adjust lower bound if exists
                if slc.lower and isinstance(slc.lower, ast.Constant) and isinstance(slc.lower.value, int):
                    slc.lower.value += 1 if random.choice([True, False]) else slc.lower.value - 1
                    mutations_made = True
                # else adjust upper bound if exists
                elif slc.upper and isinstance(slc.upper, ast.Constant) and isinstance(slc.upper.value, int):
                    slc.upper.value += 1 if random.choice([True, False]) else slc.upper.value - 1
                    mutations_made = True
        return mutations_made

    # ------------------------------------------------------------------
    # New subtle mutation strategies
    # ------------------------------------------------------------------

    def mutate_conditional_inversion(self, func_def):
        """Invert a single sub-expression inside a boolean AND/OR chain to subtly flip logic."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.BoolOp) and not mutations_made:
                # Only invert if there is at least one non-negated sub-expression.
                for idx, value in enumerate(node.values):
                    # Skip values that are already a negation (UnaryOp with Not)
                    if not (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)):
                        node.values[idx] = ast.UnaryOp(op=ast.Not(), operand=value)
                        mutations_made = True
                        break
        return mutations_made

    def mutate_range_step(self, func_def):
        """Alter the step in range() calls to change iteration granularity without obvious off-by-one errors."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and not mutations_made:
                if isinstance(node.func, ast.Name) and node.func.id == "range":
                    # If step is missing, add a subtle step (e.g., 2). If present, adjust it by ±1.
                    if len(node.args) in (1, 2):
                        node.args.append(ast.Constant(value=2))
                        mutations_made = True
                    elif len(node.args) == 3:
                        step_arg = node.args[2]
                        if isinstance(step_arg, ast.Constant) and isinstance(step_arg.value, int):
                            # Keep sign but alter magnitude by 1
                            if step_arg.value > 0:
                                step_arg.value += 1
                            elif step_arg.value < 0:
                                step_arg.value -= 1
                            else:
                                step_arg.value = 2  # previously 0, unlikely but handle
                            mutations_made = True
        return mutations_made
