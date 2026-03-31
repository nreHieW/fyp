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
        # Additional OOD mutations to diversify corruption space
        self.ood_mutations = [
            self.mutate_min_max_usage,
            self.mutate_abs_usage,
            self.mutate_append_extend,
            self.mutate_enumerate_start,
            self.mutate_break_to_continue,
            self.mutate_dict_get_default,
            self.mutate_string_lower_upper,
            self.mutate_strip_variant,
            self.mutate_join_separator,
            self.mutate_sorted_toggle_key_len,
            self.mutate_set_vs_list_cast,
            self.mutate_round_to_int,
            self.mutate_comp_filter_remove,
            self.mutate_find_vs_index,
            self.mutate_any_all_swap,
            self.mutate_zip_arg_order,
            self.mutate_len_range_endpoint,
            self.mutate_negative_indexing_shift,
            self.mutate_dict_items_variant,
            self.mutate_none_equality_operator,
        ]

    def corrupt_function(
        self,
        code,
        use_ood: str,
        max_mutations: int = 10,
    ):
        """Apply up to `max_mutations` subtle mutations to the function code.

        Returns (mutated_code: str | None, mutation_list: list[str] | str)
        """
        assert use_ood in ["both", "ood", "non_ood"]
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
            if use_ood == "both":
                mutation_order.extend(self.ood_mutations)
            elif use_ood == "ood":
                mutation_order = self.ood_mutations.copy()
            else:
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

    # ------------------------------------------------------------------
    # OOD mutation strategies (general, dataset-agnostic)
    # ------------------------------------------------------------------

    def mutate_min_max_usage(self, func_def):
        """Swap built-in min/max usage in one call."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and not mutations_made:
                if isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
                    node.func.id = "max" if node.func.id == "min" else "min"
                    mutations_made = True
                elif isinstance(node.func, ast.Attribute) and node.func.attr in ("min", "max"):
                    node.func.attr = "max" if node.func.attr == "min" else "min"
                    mutations_made = True
        return mutations_made

    def mutate_abs_usage(self, func_def):
        """Wrap an arithmetic expression in abs(), or unwrap an existing abs()."""
        mutations_made = False
        for node in ast.walk(func_def):
            if not mutations_made and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1:
                # Unwrap abs(x) -> x
                replacement = node.args[0]
                node.func.id = "__unwrap_abs__"
                node.args = [replacement]
                mutations_made = True
        if mutations_made:

            class AbsUnwrapper(ast.NodeTransformer):
                def visit_Call(self, n):
                    if isinstance(n.func, ast.Name) and n.func.id == "__unwrap_abs__" and len(n.args) == 1:
                        return n.args[0]
                    return self.generic_visit(n)

            AbsUnwrapper().visit(func_def)
            return True

        # Else try wrapping a BinOp once
        for node in ast.walk(func_def):
            if isinstance(node, ast.BinOp) and not mutations_made:
                new_call = ast.Call(func=ast.Name(id="abs", ctx=ast.Load()), args=[copy.deepcopy(node)], keywords=[])
                # Replace this BinOp in its parent by marking and second pass
                node.op = ast.Add() if isinstance(node.op, ast.Add) else node.op  # no-op touch to satisfy linter
                node.__dict__["__wrap_abs__"] = True
                mutations_made = True
                break
        if mutations_made:

            class AbsWrapper(ast.NodeTransformer):
                def visit_BinOp(self, n):
                    if getattr(n, "__wrap_abs__", False):
                        return ast.Call(func=ast.Name(id="abs", ctx=ast.Load()), args=[ast.copy_location(copy.deepcopy(n), n)], keywords=[])
                    return self.generic_visit(n)

            AbsWrapper().visit(func_def)
        return mutations_made

    def mutate_append_extend(self, func_def):
        """Toggle between list.append(x) and list.extend([x]) on one occurrence."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not mutations_made:
                if node.func.attr == "append" and len(node.args) == 1:
                    node.func.attr = "extend"
                    node.args = [ast.List(elts=[node.args[0]], ctx=ast.Load())]
                    mutations_made = True
                elif node.func.attr == "extend" and len(node.args) == 1 and isinstance(node.args[0], ast.List) and node.args[0].elts:
                    node.func.attr = "append"
                    node.args = [node.args[0].elts[0]]
                    mutations_made = True
        return mutations_made

    def mutate_enumerate_start(self, func_def):
        """Toggle enumerate start from implicit/0 to 1, or 1 to 0."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "enumerate" and not mutations_made:
                kw = next((k for k in node.keywords if k.arg == "start"), None)
                if kw and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                    kw.value.value = 0 if kw.value.value == 1 else 1
                    mutations_made = True
                else:
                    node.keywords.append(ast.keyword(arg="start", value=ast.Constant(value=1)))
                    mutations_made = True
        return mutations_made

    def mutate_break_to_continue(self, func_def):
        """Turn one Break into Continue to alter loop flow subtly."""

        class BreakToContinue(ast.NodeTransformer):
            def __init__(self):
                self.changed = False

            def visit_Break(self, n):
                if not self.changed:
                    self.changed = True
                    return ast.Continue()
                return n

        transformer = BreakToContinue()
        transformer.visit(func_def)
        return transformer.changed

    def mutate_dict_get_default(self, func_def):
        """Adjust dict.get default: 0<->1 or add default=1 if missing."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and not mutations_made:
                if len(node.args) == 1:
                    node.args.append(ast.Constant(value=1))
                    mutations_made = True
                elif len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, int):
                    node.args[1].value = 0 if node.args[1].value == 1 else 1
                    mutations_made = True
        return mutations_made

    def mutate_string_lower_upper(self, func_def):
        """Swap .lower() and .upper() on one call."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not mutations_made:
                if node.func.attr == "lower":
                    node.func.attr = "upper"
                    mutations_made = True
                elif node.func.attr == "upper":
                    node.func.attr = "lower"
                    mutations_made = True
        return mutations_made

    def mutate_strip_variant(self, func_def):
        """Change strip variant: strip<->rstrip (or lstrip if rstrip not found)."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not mutations_made:
                if node.func.attr == "strip":
                    node.func.attr = "rstrip"
                    mutations_made = True
                elif node.func.attr == "rstrip":
                    node.func.attr = "strip"
                    mutations_made = True
                elif node.func.attr == "lstrip":
                    node.func.attr = "strip"
                    mutations_made = True
        return mutations_made

    def mutate_join_separator(self, func_def):
        """Change string join separator between '', ' ', and ','."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join" and not mutations_made:
                if isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
                    sep = node.func.value.value
                    node.func.value.value = " " if sep == "" else ("" if sep == " " else ",")
                    mutations_made = True
        return mutations_made

    def mutate_sorted_toggle_key_len(self, func_def):
        """Toggle sorted key between None and len."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and not mutations_made:
                is_sorted = (isinstance(node.func, ast.Name) and node.func.id == "sorted") or (isinstance(node.func, ast.Attribute) and node.func.attr == "sort")
                if is_sorted:
                    key_kw = next((k for k in node.keywords if k.arg == "key"), None)
                    if key_kw is None:
                        node.keywords.append(ast.keyword(arg="key", value=ast.Name(id="len", ctx=ast.Load())))
                        mutations_made = True
                    else:
                        # remove key
                        node.keywords = [k for k in node.keywords if k.arg != "key"]
                        mutations_made = True
        return mutations_made

    def mutate_set_vs_list_cast(self, func_def):
        """Swap set(x) with list(x) once."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("set", "list") and not mutations_made:
                node.func.id = "list" if node.func.id == "set" else "set"
                mutations_made = True
        return mutations_made

    def mutate_round_to_int(self, func_def):
        """Change round(x) to int(x) or vice versa if found."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("round", "int") and not mutations_made:
                node.func.id = "int" if node.func.id == "round" else "round"
                mutations_made = True
        return mutations_made

    def mutate_comp_filter_remove(self, func_def):
        """Remove one filter clause from a list/set/dict comprehension."""
        mutations_made = False
        comp_types = (ast.ListComp, ast.SetComp, ast.DictComp)
        for node in ast.walk(func_def):
            if isinstance(node, comp_types) and not mutations_made:
                gens = node.generators
                for gen in gens:
                    if gen.ifs:
                        gen.ifs = gen.ifs[1:]  # drop first filter
                        mutations_made = True
                        break
        return mutations_made

    def mutate_find_vs_index(self, func_def):
        """Swap str.find and str.index usage in one place."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not mutations_made:
                if node.func.attr == "find":
                    node.func.attr = "index"
                    mutations_made = True
                elif node.func.attr == "index":
                    node.func.attr = "find"
                    mutations_made = True
        return mutations_made

    def mutate_any_all_swap(self, func_def):
        """Swap any(...) with all(...) once."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("any", "all") and not mutations_made:
                node.func.id = "all" if node.func.id == "any" else "any"
                mutations_made = True
        return mutations_made

    def mutate_zip_arg_order(self, func_def):
        """Reverse the first two positional arguments of a zip() call."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip" and len(node.args) >= 2 and not mutations_made:
                node.args[0], node.args[1] = node.args[1], node.args[0]
                mutations_made = True
        return mutations_made

    def mutate_len_range_endpoint(self, func_def):
        """Modify range(len(x)) to range(len(x)+1) or range(len(x)-1)."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and not mutations_made:
                if len(node.args) == 1 and isinstance(node.args[0], ast.Call) and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == "len":
                    len_call = node.args[0]
                    if random.choice([True, False]):
                        node.args[0] = ast.BinOp(left=len_call, op=ast.Add(), right=ast.Constant(value=1))
                    else:
                        node.args[0] = ast.BinOp(left=len_call, op=ast.Sub(), right=ast.Constant(value=1))
                    mutations_made = True
        return mutations_made

    def mutate_negative_indexing_shift(self, func_def):
        """Shift negative constant indices by one (e.g., [-1] -> [-2])."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Subscript) and not mutations_made:
                sl = node.slice
                if isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub) and isinstance(sl.operand, ast.Constant) and isinstance(sl.operand.value, int):
                    sl.operand.value += 1  # -1 -> -2, etc.
                    mutations_made = True
                elif isinstance(sl, ast.Slice):
                    # handle slice like a[-k:]
                    if sl.lower and isinstance(sl.lower, ast.UnaryOp) and isinstance(sl.lower.op, ast.USub) and isinstance(sl.lower.operand, ast.Constant) and isinstance(sl.lower.operand.value, int):
                        sl.lower.operand.value += 1
                        mutations_made = True
                    elif (
                        sl.upper and isinstance(sl.upper, ast.UnaryOp) and isinstance(sl.upper.op, ast.USub) and isinstance(sl.upper.operand, ast.Constant) and isinstance(sl.upper.operand.value, int)
                    ):
                        sl.upper.operand.value += 1
                        mutations_made = True
        return mutations_made

    def mutate_dict_items_variant(self, func_def):
        """Change dict iteration/view: items<->keys<->values for one attribute call."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("items", "keys", "values") and not mutations_made:
                if node.func.attr == "items":
                    node.func.attr = random.choice(["keys", "values"])
                elif node.func.attr == "keys":
                    node.func.attr = random.choice(["items", "values"])
                else:
                    node.func.attr = random.choice(["items", "keys"])
                mutations_made = True
        return mutations_made

    def mutate_none_equality_operator(self, func_def):
        """Toggle None comparisons between ==/!= and is/is not."""
        mutations_made = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Compare) and not mutations_made:
                # Only single comparator to avoid complex rewrites
                if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                    if len(node.ops) == 1:
                        op = node.ops[0]
                        if isinstance(op, ast.Eq):
                            node.ops[0] = ast.Is()
                            mutations_made = True
                        elif isinstance(op, ast.NotEq):
                            node.ops[0] = ast.IsNot()
                            mutations_made = True
                        elif isinstance(op, ast.Is):
                            node.ops[0] = ast.Eq()
                            mutations_made = True
                        elif isinstance(op, ast.IsNot):
                            node.ops[0] = ast.NotEq()
                            mutations_made = True
        return mutations_made
