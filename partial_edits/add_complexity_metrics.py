import argparse
import ast
import json
import os
from typing import Optional, Tuple

from partial_edits.utils.extract_utils import standardize_code_formatting


def compute_python_complexities(code: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Return (cyclomatic_complexity, cognitive_complexity) for Python code, or (None, None) if not parseable."""
    if not code:
        return None, None
    try:
        tree = ast.parse(standardize_code_formatting(code))
    except Exception:
        try:
            tree = ast.parse(code)
        except Exception:
            return None, None

    class CombinedVisitor(ast.NodeVisitor):
        def __init__(self):
            self.cc = 1
            self.cog = 0
            self.nesting = 0

        def _add_cog(self, base: int = 1):
            self.cog += base + self.nesting

        def _visit_block(self, nodes, inc_nesting: bool = True):
            if not nodes:
                return
            if inc_nesting:
                self.nesting += 1
            for child in nodes:
                self.visit(child)
            if inc_nesting:
                self.nesting -= 1

        def visit_If(self, n):
            self.cc += 1
            self._add_cog(1)
            self._visit_block(n.body, inc_nesting=True)
            if len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If):
                self.visit(n.orelse[0])
            else:
                self._visit_block(n.orelse, inc_nesting=True)

        def visit_For(self, n):
            self.cc += 1
            self._add_cog(1)
            self._visit_block(n.body, inc_nesting=True)
            self._visit_block(n.orelse, inc_nesting=True)

        def visit_AsyncFor(self, n):
            self.cc += 1
            self._add_cog(1)
            self._visit_block(n.body, inc_nesting=True)
            self._visit_block(n.orelse, inc_nesting=True)

        def visit_While(self, n):
            self.cc += 1
            self._add_cog(1)
            self._visit_block(n.body, inc_nesting=True)
            self._visit_block(n.orelse, inc_nesting=True)

        def visit_Try(self, n):
            self.cc += len(getattr(n, "handlers", []))
            if getattr(n, "finalbody", []):
                self.cc += 1
            self._visit_block(n.body, inc_nesting=True)
            for h in getattr(n, "handlers", []):
                self._add_cog(1)
                self._visit_block(h.body, inc_nesting=True)
            self._visit_block(n.orelse, inc_nesting=True)
            self._visit_block(n.finalbody, inc_nesting=True)

        def visit_With(self, n):
            self._visit_block(n.body, inc_nesting=True)

        def visit_AsyncWith(self, n):
            self._visit_block(n.body, inc_nesting=True)

        def visit_BoolOp(self, n):
            inc = max(0, len(getattr(n, "values", [])) - 1)
            self.cc += inc
            self.cog += inc
            self.generic_visit(n)

        def visit_IfExp(self, n):
            self.cc += 1
            self._add_cog(1)
            self.generic_visit(n)

        def visit_comprehension(self, n):
            self.cc += 1 + len(getattr(n, "ifs", []))
            self._add_cog(1)
            for _ in getattr(n, "ifs", []):
                self._add_cog(1)
            self.generic_visit(n)

        def visit_Assert(self, n):
            self.cc += 1
            self.generic_visit(n)

        def visit_Match(self, n):
            self.cc += len(getattr(n, "cases", []))
            for case in getattr(n, "cases", []):
                self._add_cog(1)
                self._visit_block(case.body, inc_nesting=True)
            self.generic_visit(n)

    v = CombinedVisitor()
    v.visit(tree)
    return v.cc, v.cog


def apply_complexity_metrics_to_results(results_path: str):
    if not os.path.isfile(results_path):
        raise FileNotFoundError(results_path)

    with open(results_path, "r") as f:
        data = json.load(f)

    eval_items = data.get("eval", {})

    cyclomatic_diffs_llm_vs_canonical, cyclomatic_diffs_llm_vs_corrupted, cyclomatic_diffs_canonical_vs_corrupted = [], [], []
    cyclomatic_llm_values, cyclomatic_canonical_values, cyclomatic_corrupted_values = [], [], []
    cognitive_diffs_llm_vs_canonical, cognitive_diffs_llm_vs_corrupted, cognitive_diffs_canonical_vs_corrupted = [], [], []
    cognitive_llm_values, cognitive_canonical_values, cognitive_corrupted_values = [], [], []

    for _, t in eval_items.items():
        llm, canon, corr = t.get("solution"), t.get("canonical_solution"), t.get("corrupted_solution")

        llm_cyclomatic, llm_cognitive = compute_python_complexities(llm)
        canonical_cyclomatic, canonical_cognitive = compute_python_complexities(canon)
        corrupted_cyclomatic, corrupted_cognitive = compute_python_complexities(corr)

        cyclomatic_delta_llm_vs_canonical = (llm_cyclomatic - canonical_cyclomatic) if (llm_cyclomatic is not None and canonical_cyclomatic is not None) else None
        cyclomatic_delta_llm_vs_corrupted = (llm_cyclomatic - corrupted_cyclomatic) if (llm_cyclomatic is not None and corrupted_cyclomatic is not None) else None
        cyclomatic_delta_canonical_vs_corrupted = (canonical_cyclomatic - corrupted_cyclomatic) if (canonical_cyclomatic is not None and corrupted_cyclomatic is not None) else None

        cognitive_delta_llm_vs_canonical = (llm_cognitive - canonical_cognitive) if (llm_cognitive is not None and canonical_cognitive is not None) else None
        cognitive_delta_llm_vs_corrupted = (llm_cognitive - corrupted_cognitive) if (llm_cognitive is not None and corrupted_cognitive is not None) else None
        cognitive_delta_canonical_vs_corrupted = (canonical_cognitive - corrupted_cognitive) if (canonical_cognitive is not None and corrupted_cognitive is not None) else None

        t["complexity_metrics"] = {
            "llm_complexity": llm_cyclomatic,
            "canonical_complexity": canonical_cyclomatic,
            "corrupted_complexity": corrupted_cyclomatic,
            "llm_vs_canonical_added_cyclomatic_complexity": cyclomatic_delta_llm_vs_canonical,
            "llm_vs_corrupted_added_cyclomatic_complexity": cyclomatic_delta_llm_vs_corrupted,
            "canonical_vs_corrupted_added_cyclomatic_complexity": cyclomatic_delta_canonical_vs_corrupted,
            "llm_cognitive_complexity": llm_cognitive,
            "canonical_cognitive_complexity": canonical_cognitive,
            "corrupted_cognitive_complexity": corrupted_cognitive,
            "llm_vs_canonical_added_cognitive_complexity": cognitive_delta_llm_vs_canonical,
            "llm_vs_corrupted_added_cognitive_complexity": cognitive_delta_llm_vs_corrupted,
            "canonical_vs_corrupted_added_cognitive_complexity": cognitive_delta_canonical_vs_corrupted,
        }

        if llm_cyclomatic is not None:
            cyclomatic_llm_values.append(llm_cyclomatic)
        if canonical_cyclomatic is not None:
            cyclomatic_canonical_values.append(canonical_cyclomatic)
        if corrupted_cyclomatic is not None:
            cyclomatic_corrupted_values.append(corrupted_cyclomatic)

        if llm_cognitive is not None:
            cognitive_llm_values.append(llm_cognitive)
        if canonical_cognitive is not None:
            cognitive_canonical_values.append(canonical_cognitive)
        if corrupted_cognitive is not None:
            cognitive_corrupted_values.append(corrupted_cognitive)
        cyclomatic_diffs_llm_vs_canonical.append(cyclomatic_delta_llm_vs_canonical)
        cyclomatic_diffs_llm_vs_corrupted.append(cyclomatic_delta_llm_vs_corrupted)
        cyclomatic_diffs_canonical_vs_corrupted.append(cyclomatic_delta_canonical_vs_corrupted)
        cognitive_diffs_llm_vs_canonical.append(cognitive_delta_llm_vs_canonical)
        cognitive_diffs_llm_vs_corrupted.append(cognitive_delta_llm_vs_corrupted)
        cognitive_diffs_canonical_vs_corrupted.append(cognitive_delta_canonical_vs_corrupted)

    data.setdefault("metrics", {})
    cyclomatic_deltas_llm_vs_canonical = [v for v in cyclomatic_diffs_llm_vs_canonical if v is not None]
    cyclomatic_deltas_llm_vs_corrupted = [v for v in cyclomatic_diffs_llm_vs_corrupted if v is not None]
    cyclomatic_deltas_canonical_vs_corrupted = [v for v in cyclomatic_diffs_canonical_vs_corrupted if v is not None]
    cognitive_deltas_llm_vs_canonical = [v for v in cognitive_diffs_llm_vs_canonical if v is not None]
    cognitive_deltas_llm_vs_corrupted = [v for v in cognitive_diffs_llm_vs_corrupted if v is not None]
    cognitive_deltas_canonical_vs_corrupted = [v for v in cognitive_diffs_canonical_vs_corrupted if v is not None]

    avg_added_cyclomatic_llm_vs_canonical = (sum(cyclomatic_deltas_llm_vs_canonical) / len(cyclomatic_deltas_llm_vs_canonical)) if cyclomatic_deltas_llm_vs_canonical else None
    avg_added_cyclomatic_llm_vs_corrupted = (sum(cyclomatic_deltas_llm_vs_corrupted) / len(cyclomatic_deltas_llm_vs_corrupted)) if cyclomatic_deltas_llm_vs_corrupted else None
    avg_added_cyclomatic_canonical_vs_corrupted = (sum(cyclomatic_deltas_canonical_vs_corrupted) / len(cyclomatic_deltas_canonical_vs_corrupted)) if cyclomatic_deltas_canonical_vs_corrupted else None
    avg_added_cognitive_llm_vs_canonical = (sum(cognitive_deltas_llm_vs_canonical) / len(cognitive_deltas_llm_vs_canonical)) if cognitive_deltas_llm_vs_canonical else None
    avg_added_cognitive_llm_vs_corrupted = (sum(cognitive_deltas_llm_vs_corrupted) / len(cognitive_deltas_llm_vs_corrupted)) if cognitive_deltas_llm_vs_corrupted else None
    avg_added_cognitive_canonical_vs_corrupted = (sum(cognitive_deltas_canonical_vs_corrupted) / len(cognitive_deltas_canonical_vs_corrupted)) if cognitive_deltas_canonical_vs_corrupted else None

    avg_cyclomatic_llm = (sum(cyclomatic_llm_values) / len(cyclomatic_llm_values)) if cyclomatic_llm_values else None
    avg_cyclomatic_canonical = (sum(cyclomatic_canonical_values) / len(cyclomatic_canonical_values)) if cyclomatic_canonical_values else None
    avg_cyclomatic_corrupted = (sum(cyclomatic_corrupted_values) / len(cyclomatic_corrupted_values)) if cyclomatic_corrupted_values else None
    avg_cognitive_llm = (sum(cognitive_llm_values) / len(cognitive_llm_values)) if cognitive_llm_values else None
    avg_cognitive_canonical = (sum(cognitive_canonical_values) / len(cognitive_canonical_values)) if cognitive_canonical_values else None
    avg_cognitive_corrupted = (sum(cognitive_corrupted_values) / len(cognitive_corrupted_values)) if cognitive_corrupted_values else None

    data["metrics"]["complexity_metrics"] = {
        "avg_added_llm_vs_canonical": avg_added_cyclomatic_llm_vs_canonical,
        "avg_added_llm_vs_corrupted": avg_added_cyclomatic_llm_vs_corrupted,
        "avg_added_canonical_vs_corrupted": avg_added_cyclomatic_canonical_vs_corrupted,
        "avg_llm_complexity": avg_cyclomatic_llm,
        "avg_canonical_complexity": avg_cyclomatic_canonical,
        "avg_corrupted_complexity": avg_cyclomatic_corrupted,
        "avg_added_llm_vs_canonical_cognitive": avg_added_cognitive_llm_vs_canonical,
        "avg_added_llm_vs_corrupted_cognitive": avg_added_cognitive_llm_vs_corrupted,
        "avg_added_canonical_vs_corrupted_cognitive": avg_added_cognitive_canonical_vs_corrupted,
        "avg_llm_cognitive_complexity": avg_cognitive_llm,
        "avg_canonical_cognitive_complexity": avg_cognitive_canonical,
        "avg_corrupted_cognitive_complexity": avg_cognitive_corrupted,
    }

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def main():
    p = argparse.ArgumentParser(description="Add cyclomatic and cognitive complexity metrics to an evaluation JSON (Python-only).")
    p.add_argument("--results_path", required=True)
    args = p.parse_args()
    apply_complexity_metrics_to_results(args.results_path)
    print("Updated:", args.results_path)


if __name__ == "__main__":
    main()
