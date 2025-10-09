import ast
import Levenshtein
import tokenize
import io
from nltk.translate.chrf_score import corpus_chrf
from codebleu import calc_codebleu
from functools import wraps
from typing import Optional, Tuple


def standardize_code_formatting(code: str) -> str:
    """
    Strip comments and normalize formatting from Python code using AST.
    This removes comments, extra newlines, and standardizes formatting.

    Args:
        code: Python code string

    Returns:
        Code string with comments removed and formatting normalized
    """
    if not code.strip():
        return code

    try:
        # Use AST to parse and unparse - this removes comments and normalizes formatting
        tree = ast.parse(code)
        return ast.unparse(tree)
    except Exception:
        # Fallback to tokenize approach for comment removal only
        try:
            tokens = []
            readline = io.StringIO(code).readline

            for token in tokenize.generate_tokens(readline):
                # Skip comment tokens but keep everything else
                if token.type != tokenize.COMMENT:
                    tokens.append(token)

            # Reconstruct the code without comments
            return tokenize.untokenize(tokens)
        except Exception:
            # Final fallback: simple line-by-line processing
            lines = code.splitlines()
            result_lines = []

            for line in lines:
                # Find # that's not inside a string
                in_string = False
                quote_char = None
                i = 0

                while i < len(line):
                    char = line[i]

                    if not in_string and char in ['"', "'"]:
                        # Check for triple quotes
                        if i + 2 < len(line) and line[i : i + 3] in ['"""', "'''"]:
                            in_string = True
                            quote_char = line[i : i + 3]
                            i += 3
                            continue
                        else:
                            in_string = True
                            quote_char = char
                    elif in_string and char == quote_char[0]:
                        if len(quote_char) == 3:
                            if i + 2 < len(line) and line[i : i + 3] == quote_char:
                                in_string = False
                                quote_char = None
                                i += 3
                                continue
                        else:
                            if i == 0 or line[i - 1] != "\\":
                                in_string = False
                                quote_char = None
                    elif not in_string and char == "#":
                        # Found comment, truncate line here
                        line = line[:i].rstrip()
                        break

                    i += 1

                result_lines.append(line)

            return "\n".join(result_lines)


def tokenize_code(code: str):
    try:
        tokens = []
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in stream:
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT):
                continue
            tokens.append(tok.string)
        return tokens
    except Exception as e:
        return code.split()


def handle_comments(func):
    @wraps(func)
    def wrapper(*args, ignore_comments=False, **kwargs):
        if ignore_comments and len(args) >= 2:
            args = list(args)
            args[0] = standardize_code_formatting(args[0])
            args[1] = standardize_code_formatting(args[1])
        return func(*args, **kwargs)

    return wrapper


@handle_comments
def get_levenshtein_distance(reference: str, prediction: str, normalize: bool = False) -> int:
    ref_tokens = tokenize_code(reference)
    pred_tokens = tokenize_code(prediction)
    distance = Levenshtein.distance(ref_tokens, pred_tokens)
    if normalize:
        return distance / max(len(ref_tokens), len(pred_tokens))
    return distance


@handle_comments
def get_chrf_score(reference: str, prediction: str) -> float:
    ref_sentences = [reference] if reference.strip() else [""]
    pred_sentences = [prediction] if prediction.strip() else [""]
    return corpus_chrf(ref_sentences, pred_sentences)


# codebleu has a strip inside which affects the case with only the function body so we cannot use tokenize_code
@handle_comments
def get_codebleu_score(reference: str, prediction: str, lang="python", weights=(1 / 3, 1 / 3, 1 / 3, 0)) -> dict:
    result = calc_codebleu([reference], [prediction], lang=lang, weights=weights, tokenizer=None)
    return {"codebleu": result.get("codebleu", 0.0)}


def calculate_corpus_codebleu_score(references: list, predictions: list, lang="python", weights=(1 / 3, 1 / 3, 1 / 3, 0), ignore_comments: bool = False) -> float:
    if ignore_comments:
        references = [standardize_code_formatting(ref) for ref in references]
        predictions = [standardize_code_formatting(pred) for pred in predictions]

    result = calc_codebleu(references, predictions, lang=lang, weights=weights, tokenizer=None)
    return result.get("codebleu", 0.0)


def compute_python_complexities(code: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
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

    visitor = CombinedVisitor()
    visitor.visit(tree)
    return visitor.cc, visitor.cog


def get_cognitive_complexity_similarity(reference: str, prediction: str) -> float:
    _, ref_cognitive = compute_python_complexities(reference)
    _, pred_cognitive = compute_python_complexities(prediction)

    if ref_cognitive is None or pred_cognitive is None:
        return 0.0

    diff = abs(ref_cognitive - pred_cognitive)
    return 1.0 / (1.0 + diff)
