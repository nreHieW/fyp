import difflib
import Levenshtein
import subprocess
import tempfile
import os
import tokenize
import io
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import corpus_chrf
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from codebleu import calc_codebleu
from functools import wraps

from .extract_utils import standardize_code_formatting


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


def call_diffsitter(reference: str, prediction: str) -> dict:
    try:
        # Create temporary files for the two code texts
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f1, tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f2:

            f1.write(reference)
            f1.flush()
            f2.write(prediction)
            f2.flush()

            # Call diffsitter with the temporary files
            cmd = ["diffsitter", "--file-type", "python", f1.name, f2.name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Clean up temporary files
            os.unlink(f1.name)
            os.unlink(f2.name)

            if result.returncode == 0:
                return {"success": True, "diff_output": result.stdout, "has_differences": bool(result.stdout.strip()), "error": None}
            else:
                return {"success": False, "diff_output": None, "has_differences": False, "error": f"diffsitter error: {result.stderr}"}

    except subprocess.TimeoutExpired:
        return {"success": False, "diff_output": None, "has_differences": False, "error": "diffsitter timed out"}
    except Exception as e:
        return {"success": False, "diff_output": None, "has_differences": False, "error": f"Error calling diffsitter: {str(e)}"}


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
def count_diff_lines(reference: str, prediction: str) -> int:
    ref_lines = reference.splitlines()
    pred_lines = prediction.splitlines()

    m, n = len(ref_lines), len(pred_lines)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_lines[i - 1].strip() == pred_lines[j - 1].strip():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


@handle_comments
def get_levenshtein_distance(reference: str, prediction: str) -> int:
    ref_tokens = tokenize_code(reference)
    pred_tokens = tokenize_code(prediction)
    return Levenshtein.distance(ref_tokens, pred_tokens)


@handle_comments
def get_structured_diff(reference: str, prediction: str) -> list:

    if not reference and not prediction:
        return []

    ref_lines = reference.splitlines(keepends=True) if reference else []
    pred_lines = prediction.splitlines(keepends=True) if prediction else []

    if ref_lines == pred_lines:
        return [{"type": "context", "content": reference}]

    if len(ref_lines) > 1000 or len(pred_lines) > 1000:
        return [{"type": "summary", "content": f"Files too large for diff. Reference: {len(ref_lines)} lines, Prediction: {len(pred_lines)} lines"}]

    # Traditional difflib-based diff (fallback or when use_diffsitter=False)
    diff_lines = list(difflib.unified_diff(ref_lines, pred_lines, lineterm=""))

    result = []

    for line in diff_lines:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        elif line.startswith("+"):
            result.append({"type": "added", "content": line[1:].rstrip()})
        elif line.startswith("-"):
            result.append({"type": "removed", "content": line[1:].rstrip()})
        else:
            result.append({"type": "context", "content": line.rstrip()})

    return result


@handle_comments
def get_rouge_scores(reference: str, prediction: str, rouge_types=None) -> dict:

    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, prediction)

    result = {}
    for rouge_type in rouge_types:
        score = scores[rouge_type]
        result[f"{rouge_type}_fmeasure"] = score.fmeasure

    return result


@handle_comments
def get_meteor_score(reference: str, prediction: str) -> float:
    ref_tokens = word_tokenize(reference.lower())
    pred_tokens = word_tokenize(prediction.lower())
    return meteor_score([ref_tokens], pred_tokens)


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


@handle_comments
def calculate_bleu_score(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_code(reference)
    pred_tokens = tokenize_code(prediction)
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)


def calculate_corpus_bleu_score(references: list, predictions: list, ignore_comments: bool = False) -> float:
    if ignore_comments:
        references = [standardize_code_formatting(ref) for ref in references]
        predictions = [standardize_code_formatting(pred) for pred in predictions]

    tokenized_refs = []
    tokenized_preds = []
    for ref, pred in zip(references, predictions):
        ref_tokens = tokenize_code(ref)
        pred_tokens = tokenize_code(pred)
        tokenized_refs.append([ref_tokens])
        tokenized_preds.append(pred_tokens)

    return corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=SmoothingFunction().method1)


def calculate_corpus_codebleu_score(references: list, predictions: list, lang="python", weights=(1 / 3, 1 / 3, 1 / 3, 0), ignore_comments: bool = False) -> float:
    if ignore_comments:
        references = [standardize_code_formatting(ref) for ref in references]
        predictions = [standardize_code_formatting(pred) for pred in predictions]

    result = calc_codebleu(references, predictions, lang=lang, weights=weights, tokenizer=None)
    return result.get("codebleu", 0.0)


@handle_comments
def get_diffsitter_edit_distance(reference: str, prediction: str) -> int:
    diffsitter_result = call_diffsitter(reference, prediction)

    if not diffsitter_result["success"]:
        # If diffsitter fails, return -1 to indicate failure/unavailable
        return -1

    if not diffsitter_result["has_differences"]:
        return 0

    diff_output = diffsitter_result["diff_output"]

    lines_changed = set()
    for line in diff_output.split("\n"):
        line = line.strip()
        line = line.replace("\x1b[31m", "").replace("\x1b[0m", "").replace("\x1b[1m", "").replace("\x1b[32m", "")
        if line and line[0].isdigit():
            prefix = line.split(":")[0].strip()
            if "-" in prefix:  # this is a range
                first, last = prefix.split("-")
                r = range(int(first), int(last) + 1)
                [lines_changed.add(i) for i in r]
            else:
                if prefix.isdigit():
                    lines_changed.add(int(prefix))

    return len(lines_changed)
