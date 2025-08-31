import ast
import Levenshtein
import tokenize
import io
from nltk.translate.chrf_score import corpus_chrf
from codebleu import calc_codebleu
from functools import wraps


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
