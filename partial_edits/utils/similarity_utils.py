import difflib
import Levenshtein
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import corpus_chrf
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from codebleu import calc_codebleu


from .extract_utils import standardize_code_formatting


def count_diff_lines(original: str, corrupted: str, ignore_comments: bool = False) -> int:
    """Count how many lines are different between two code snippets using difflib."""
    if ignore_comments:
        original = standardize_code_formatting(original)
        corrupted = standardize_code_formatting(corrupted)

    orig_lines = original.splitlines(keepends=True)
    corr_lines = corrupted.splitlines(keepends=True)

    # Use unified_diff to get the actual changes
    diff = list(difflib.unified_diff(orig_lines, corr_lines, lineterm=""))

    # Count lines that are additions or deletions (start with + or -)
    # Skip the first 3 lines which are headers (@@ lines)
    changed_lines = 0
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            changed_lines += 1
        elif line.startswith("-") and not line.startswith("---"):
            changed_lines += 1

    # Since we count both additions and deletions, divide by 2 for line replacements
    # But handle pure additions/deletions correctly
    additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    # Return the maximum of additions or deletions (representing lines changed)
    return max(additions, deletions)


def get_levenshtein_distance(original: str, corrupted: str, ignore_comments: bool = False) -> int:
    """Get the Levenshtein distance between two code snippets."""
    if ignore_comments:
        original = standardize_code_formatting(original)
        corrupted = standardize_code_formatting(corrupted)

    return Levenshtein.distance(original, corrupted)


def get_structured_diff(text1: str, text2: str, ignore_comments: bool = False) -> list:
    """Get structured diff data between two text strings using difflib."""
    if ignore_comments:
        text1 = standardize_code_formatting(text1)
        text2 = standardize_code_formatting(text2)

    if not text1 and not text2:
        return []

    lines1 = text1.splitlines(keepends=True) if text1 else []
    lines2 = text2.splitlines(keepends=True) if text2 else []

    if lines1 == lines2:
        return [{"type": "context", "content": text1}]

    if len(lines1) > 1000 or len(lines2) > 1000:
        return [{"type": "summary", "content": f"Files too large for diff. Text1: {len(lines1)} lines, Text2: {len(lines2)} lines"}]

    diff_lines = list(difflib.unified_diff(lines1, lines2, lineterm=""))

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


def get_rouge_scores(reference: str, prediction: str, rouge_types=None, ignore_comments: bool = False) -> dict:
    """Get ROUGE scores between reference and prediction text."""
    if ignore_comments:
        reference = standardize_code_formatting(reference)
        prediction = standardize_code_formatting(prediction)

    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, prediction)

    # Only return F-measure for each ROUGE type
    result = {}
    for rouge_type in rouge_types:
        score = scores[rouge_type]
        result[f"{rouge_type}_fmeasure"] = score.fmeasure

    return result


def get_meteor_score(reference: str, prediction: str, ignore_comments: bool = False) -> float:
    """Get METEOR score between reference and prediction text."""
    if ignore_comments:
        reference = standardize_code_formatting(reference)
        prediction = standardize_code_formatting(prediction)

    try:
        # Tokenize the texts
        ref_tokens = word_tokenize(reference.lower())
        pred_tokens = word_tokenize(prediction.lower())

        # Calculate METEOR score
        score = meteor_score([ref_tokens], pred_tokens)
        return score
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0


def get_chrf_score(reference: str, prediction: str, ignore_comments: bool = False) -> float:
    """Get chrF score between reference and prediction text."""
    if ignore_comments:
        reference = standardize_code_formatting(reference)
        prediction = standardize_code_formatting(prediction)

    try:
        # chrF works at character level, so we split by lines or use the text as-is
        # Convert to list of sentences (lines) as expected by corpus_chrf
        ref_sentences = [reference] if reference.strip() else [""]
        pred_sentences = [prediction] if prediction.strip() else [""]

        # Calculate chrF score
        score = corpus_chrf(ref_sentences, pred_sentences)
        return score
    except Exception as e:
        print(f"Error calculating chrF score: {e}")
        return 0.0


# https://github.com/microsoft/CodeXGLUE/issues/46
def get_codebleu_score(reference: str, prediction: str, lang="python", weights=(1 / 3, 1 / 3, 1 / 3, 0), ignore_comments: bool = False) -> dict:
    """Get CodeBLEU score between reference and prediction code."""
    if ignore_comments:
        reference = standardize_code_formatting(reference)
        prediction = standardize_code_formatting(prediction)

    try:
        result = calc_codebleu([reference], [prediction], lang=lang, weights=weights, tokenizer=None)
        return {"codebleu": result.get("codebleu", 0.0)}
    except Exception as e:
        print(f"Error calculating CodeBLEU score: {e}")
        return {"codebleu": 0.0}


def calculate_bleu_score(original: str, edited: str, ignore_comments: bool = False) -> float:
    """Calculate BLEU score between two code snippets"""
    if ignore_comments:
        original = standardize_code_formatting(original)
        edited = standardize_code_formatting(edited)

    tokens1 = original.replace("\n", " ").replace("\t", " ").split()
    tokens2 = edited.replace("\n", " ").replace("\t", " ").split()

    smoothing = SmoothingFunction().method1

    try:
        bleu_score = sentence_bleu([tokens1], tokens2, smoothing_function=smoothing)
        return bleu_score
    except:
        return 0.0


def get_all_similarity_metrics(reference: str, prediction: str, lang="python", ignore_comments: bool = False) -> dict:
    """Get all similarity metrics between reference and prediction text/code."""
    metrics = {
        "bleu_score": calculate_bleu_score(reference, prediction, ignore_comments=ignore_comments),
        "levenshtein_distance": get_levenshtein_distance(reference, prediction, ignore_comments=ignore_comments),
        "edit_distance": count_diff_lines(reference, prediction, ignore_comments=ignore_comments),
        "rouge_scores": get_rouge_scores(reference, prediction, ignore_comments=ignore_comments),
        "meteor_score": get_meteor_score(reference, prediction, ignore_comments=ignore_comments),
        "chrf_score": get_chrf_score(reference, prediction, ignore_comments=ignore_comments),
        "codebleu_scores": get_codebleu_score(reference, prediction, lang=lang, ignore_comments=ignore_comments),
    }

    return metrics


def calculate_corpus_bleu_score(references: list, predictions: list, ignore_comments: bool = False) -> float:
    """Calculate corpus-level BLEU score between lists of references and predictions"""
    if ignore_comments:
        references = [standardize_code_formatting(ref) for ref in references]
        predictions = [standardize_code_formatting(pred) for pred in predictions]

    # Tokenize all references and predictions
    tokenized_refs = []
    tokenized_preds = []

    for ref, pred in zip(references, predictions):
        ref_tokens = ref.replace("\n", " ").replace("\t", " ").split()
        pred_tokens = pred.replace("\n", " ").replace("\t", " ").split()
        tokenized_refs.append([ref_tokens])  # Wrap each reference in a list
        tokenized_preds.append(pred_tokens)

    smoothing = SmoothingFunction().method1

    try:
        bleu_score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothing)
        return bleu_score
    except Exception as e:
        print(f"Error calculating corpus BLEU score: {e}")
        return 0.0


def calculate_corpus_codebleu_score(references: list, predictions: list, lang="python", weights=(1 / 3, 1 / 3, 1 / 3, 0), ignore_comments: bool = False) -> float:
    """Calculate corpus-level CodeBLEU score between lists of references and predictions"""
    if ignore_comments:
        references = [standardize_code_formatting(ref) for ref in references]
        predictions = [standardize_code_formatting(pred) for pred in predictions]

    try:
        result = calc_codebleu(references, predictions, lang=lang, weights=weights, tokenizer=None)
        return result.get("codebleu", 0.0)
    except Exception as e:
        print(f"Error calculating corpus CodeBLEU score: {e}")
        return 0.0
