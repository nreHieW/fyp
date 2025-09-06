# deepcoder-minimal-edit

> Replace the placeholders below, then remove this callout. Keep the Evaluation Reports section at the bottom intact so reports can auto-render.

### Overview
- **Environment ID**: `deepcoder-minimal-edit`
- **Short description**: <one-sentence description>
- **Tags**: <comma-separated tags>

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: <train/eval counts>

### Task
- **Type**: <single-turn | multi-turn | tool use>
- **Parser**: <e.g., ThinkParser, XMLParser, custom>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval deepcoder-minimal-edit
```

Configure model and sampling:

```bash
uv run vf-eval deepcoder-minimal-edit   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

### Mutations
- mutate_comparison_operators: Swap comparison operators (e.g., <↔<=, ==↔!=, >↔>=) in one compare.
- mutate_range_bounds: Introduce ±1 off-by-one in range() bounds (start/end or single-arg).
- mutate_sort_order: Flip sorting direction (reverse/ascending) or insert a toggled kwarg.
- mutate_accumulator_init: Change common accumulator inits (0→1, []→[0]) once.
- mutate_arithmetic_operators: Swap basic arithmetic ops (+↔-, *↔//, //↔/).
- mutate_edge_case_guards: Loosen/tweak edge-case guards (invert not, bump 0/1 checks).
- mutate_list_indexing: Shift an index by one or remove a ±1 in computed indices.
- mutate_function_call_name: Replace mean→median, max→min, sum→len in a single call.
- mutate_remove_copy_calls: Remove `.copy()` to alias original reference.
- mutate_boolean_constants: Flip a boolean literal True↔False.
- mutate_numeric_constants: Perturb a numeric literal (int ±1 except 0/1; float ±0.1).
- mutate_slice_bounds: Shift a slice lower/upper integer boundary by one.
- mutate_conditional_inversion: Negate one sub-expression in an AND/OR chain.
- mutate_range_step: Add or subtly adjust the step argument in range().
- mutate_min_max_usage (OOD): Swap built-in min and max in one usage.
- mutate_abs_usage (OOD): Wrap a BinOp in abs() or unwrap an existing abs().
- mutate_append_extend (OOD): Toggle list.append(x) ↔ list.extend([x]).
- mutate_enumerate_start (OOD): Toggle/add enumerate(start) between 0 and 1.
- mutate_break_to_continue (OOD): Convert one break to continue to alter loop flow.
- mutate_dict_get_default (OOD): Toggle/add dict.get default (0↔1; add default=1).
- mutate_string_lower_upper (OOD): Swap string case methods .lower() ↔ .upper().
- mutate_strip_variant (OOD): Switch strip variants strip ↔ rstrip/lstrip.
- mutate_join_separator (OOD): Change join separator among "", " ", and ",".
- mutate_sorted_toggle_key_len (OOD): Toggle presence of key=len (add/remove) in sort/sorted.
- mutate_set_vs_list_cast (OOD): Switch set(x) ↔ list(x) in one call.
- mutate_round_to_int (OOD): Swap round(x) ↔ int(x).
- mutate_comp_filter_remove (OOD): Remove one filter from a comprehension.
- mutate_find_vs_index (OOD): Swap str.find ↔ str.index in one location.
- mutate_any_all_swap (OOD): Swap any(...) ↔ all(...).
- mutate_zip_arg_order (OOD): Swap the first two arguments of zip().
- mutate_len_range_endpoint (OOD): Change range(len(x)) to len(x)±1.
- mutate_negative_indexing_shift (OOD): Shift a negative constant index by one (e.g., -1→-2).
- mutate_dict_items_variant (OOD): Toggle dict view calls among items/keys/values.
- mutate_none_equality_operator (OOD): Toggle None comparisons between ==/!= and is/is not.
