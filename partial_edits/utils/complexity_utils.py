import ast
from typing import Dict, Iterable, Optional, Tuple

from .extract_utils import standardize_code_formatting


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


def compute_per_task_complexity(llm: Optional[str], canonical: Optional[str], corrupted: Optional[str]) -> Dict[str, Optional[int]]:
    llm_cyclomatic, llm_cognitive = compute_python_complexities(llm)
    canonical_cyclomatic, canonical_cognitive = compute_python_complexities(canonical)
    corrupted_cyclomatic, corrupted_cognitive = compute_python_complexities(corrupted)

    cyclomatic_delta_llm_vs_canonical = (llm_cyclomatic - canonical_cyclomatic) if (llm_cyclomatic is not None and canonical_cyclomatic is not None) else None
    cyclomatic_delta_llm_vs_corrupted = (llm_cyclomatic - corrupted_cyclomatic) if (llm_cyclomatic is not None and corrupted_cyclomatic is not None) else None
    cyclomatic_delta_canonical_vs_corrupted = (canonical_cyclomatic - corrupted_cyclomatic) if (canonical_cyclomatic is not None and corrupted_cyclomatic is not None) else None

    cognitive_delta_llm_vs_canonical = (llm_cognitive - canonical_cognitive) if (llm_cognitive is not None and canonical_cognitive is not None) else None
    cognitive_delta_llm_vs_corrupted = (llm_cognitive - corrupted_cognitive) if (llm_cognitive is not None and corrupted_cognitive is not None) else None
    cognitive_delta_canonical_vs_corrupted = (canonical_cognitive - corrupted_cognitive) if (canonical_cognitive is not None and corrupted_cognitive is not None) else None

    return {
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


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_complexity_metrics(per_task_metrics: Iterable[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    per_task_metrics = list(per_task_metrics)
    if not per_task_metrics:
        return {
            "avg_added_llm_vs_canonical": None,
            "avg_added_llm_vs_corrupted": None,
            "avg_added_canonical_vs_corrupted": None,
            "avg_llm_complexity": None,
            "avg_canonical_complexity": None,
            "avg_corrupted_complexity": None,
            "avg_added_llm_vs_canonical_cognitive": None,
            "avg_added_llm_vs_corrupted_cognitive": None,
            "avg_added_canonical_vs_corrupted_cognitive": None,
            "avg_llm_cognitive_complexity": None,
            "avg_canonical_cognitive_complexity": None,
            "avg_corrupted_cognitive_complexity": None,
        }

    cyclomatic_llm_values = [m.get("llm_complexity") for m in per_task_metrics]
    cyclomatic_canonical_values = [m.get("canonical_complexity") for m in per_task_metrics]
    cyclomatic_corrupted_values = [m.get("corrupted_complexity") for m in per_task_metrics]
    cyclomatic_diffs_llm_vs_canonical = [m.get("llm_vs_canonical_added_cyclomatic_complexity") for m in per_task_metrics]
    cyclomatic_diffs_llm_vs_corrupted = [m.get("llm_vs_corrupted_added_cyclomatic_complexity") for m in per_task_metrics]
    cyclomatic_diffs_canonical_vs_corrupted = [m.get("canonical_vs_corrupted_added_cyclomatic_complexity") for m in per_task_metrics]

    cognitive_llm_values = [m.get("llm_cognitive_complexity") for m in per_task_metrics]
    cognitive_canonical_values = [m.get("canonical_cognitive_complexity") for m in per_task_metrics]
    cognitive_corrupted_values = [m.get("corrupted_cognitive_complexity") for m in per_task_metrics]
    cognitive_diffs_llm_vs_canonical = [m.get("llm_vs_canonical_added_cognitive_complexity") for m in per_task_metrics]
    cognitive_diffs_llm_vs_corrupted = [m.get("llm_vs_corrupted_added_cognitive_complexity") for m in per_task_metrics]
    cognitive_diffs_canonical_vs_corrupted = [m.get("canonical_vs_corrupted_added_cognitive_complexity") for m in per_task_metrics]

    return {
        "avg_added_llm_vs_canonical": _mean_or_none(cyclomatic_diffs_llm_vs_canonical),
        "avg_added_llm_vs_corrupted": _mean_or_none(cyclomatic_diffs_llm_vs_corrupted),
        "avg_added_canonical_vs_corrupted": _mean_or_none(cyclomatic_diffs_canonical_vs_corrupted),
        "avg_llm_complexity": _mean_or_none(cyclomatic_llm_values),
        "avg_canonical_complexity": _mean_or_none(cyclomatic_canonical_values),
        "avg_corrupted_complexity": _mean_or_none(cyclomatic_corrupted_values),
        "avg_added_llm_vs_canonical_cognitive": _mean_or_none(cognitive_diffs_llm_vs_canonical),
        "avg_added_llm_vs_corrupted_cognitive": _mean_or_none(cognitive_diffs_llm_vs_corrupted),
        "avg_added_canonical_vs_corrupted_cognitive": _mean_or_none(cognitive_diffs_canonical_vs_corrupted),
        "avg_llm_cognitive_complexity": _mean_or_none(cognitive_llm_values),
        "avg_canonical_cognitive_complexity": _mean_or_none(cognitive_canonical_values),
        "avg_corrupted_cognitive_complexity": _mean_or_none(cognitive_corrupted_values),
    }
