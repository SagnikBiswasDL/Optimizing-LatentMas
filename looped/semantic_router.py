"""Frozen question features and the per-depth quality, cost, and depth rule.

Sentence-transformers and torch are imported only inside the embedding
functions. Logistic and ridge heads see question text, never pilot labels
from a held-out question, and never generated answers.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

CHUNKING_VERSION = "windows-v1"
PROB_CLIP = 1e-6
COST_FLOOR = 1.0


class ConstantProbability:
    """Smoothed training frequency used when a depth has one observed class."""

    def __init__(self, probability: float):
        self.probability = float(probability)
        self.kind = "constant"

    def predict(self, X) -> np.ndarray:
        return np.full(X.shape[0], self.probability, dtype=np.float64)


class ConstantCost:
    """Per-depth training-mean cost, in the scaled units the ridge head uses."""

    def __init__(self, value: float):
        self.value = float(value)
        self.kind = "constant"

    def predict(self, X) -> np.ndarray:
        return np.full(X.shape[0], self.value, dtype=np.float64)


def clip_probability(p: np.ndarray, eps: float = PROB_CLIP) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def length_features(texts: Sequence[str]) -> np.ndarray:
    """``log(1 + characters)`` and ``log(1 + whitespace-separated words)``."""
    chars = np.array([len(text) for text in texts], dtype=np.float64)
    words = np.array([len(text.split()) for text in texts], dtype=np.float64)
    return np.column_stack([np.log1p(chars), np.log1p(words)])


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("embedding has zero or nonfinite norm")
    return vector / norm


def chunk_content_ids(content_ids: Sequence[int], window_size: int) -> List[List[int]]:
    """Consecutive non-overlapping windows. The concatenation is the original id list."""
    if window_size < 1:
        raise ValueError("window_size must leave room for at least one content token")
    ids = [int(token) for token in content_ids]
    if len(ids) <= window_size:
        return [ids]
    return [ids[start:start + window_size] for start in range(0, len(ids), window_size)]


def combine_window_vectors(pooled: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """L2-normalize each window, average by non-special token count, and return that average.

    The caller L2-normalizes the returned vector. ``pooled`` is the sentence-transformer
    pooling output, before that normalization.
    """
    pooled = np.asarray(pooled, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if pooled.ndim != 2 or len(weights) != len(pooled):
        raise ValueError("window vectors and weights do not match")
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("window weights must be positive and finite")
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if np.any(norms < 1e-12) or not np.isfinite(norms).all():
        raise ValueError("a window embedding has zero or nonfinite norm")
    unit = pooled / norms
    return (unit * weights[:, None]).sum(axis=0) / weights.sum()


def cache_fingerprint(*, encoder: str, revision: str, text_hash: str, max_seq_length: int,
                      tokenizer: str) -> str:
    payload = {
        "chunking": CHUNKING_VERSION,
        "encoder": encoder,
        "max_seq_length": int(max_seq_length),
        "normalize": "l2",
        "precision": "float32",
        "revision": revision,
        "text_sha256": text_hash,
        "tokenizer": tokenizer,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def loop_tokens_from_scaled(scaled: np.ndarray, unit: float) -> Tuple[np.ndarray, np.ndarray]:
    """Map a scaled cost prediction back to loop-tokens and floor it at 1."""
    raw = np.asarray(scaled, dtype=np.float64) * float(unit)
    hit = raw < COST_FLOOR
    return np.maximum(raw, COST_FLOOR), hit


def choose_depth(p_correct: np.ndarray, expected_loop_tokens: np.ndarray, lambda_: float,
                 depths: Sequence[int], unit: float = 10000.0) -> np.ndarray:
    """Column index maximizing predicted correctness minus lambda times scaled cost.

    Ties break toward lower predicted cost, then the smaller numerical depth.
    """
    p_correct = np.asarray(p_correct, dtype=np.float64)
    expected_loop_tokens = np.asarray(expected_loop_tokens, dtype=np.float64)
    if p_correct.shape != expected_loop_tokens.shape:
        raise ValueError("quality and cost predictions have different shapes")
    if p_correct.ndim != 2 or p_correct.shape[1] != len(depths):
        raise ValueError("prediction width does not match the action list")
    utility = p_correct - float(lambda_) * expected_loop_tokens / float(unit)
    depth_values = np.broadcast_to(np.asarray(depths, dtype=np.float64), utility.shape)
    order = np.lexsort((depth_values, expected_loop_tokens, -utility))
    return order[:, 0].astype(np.int64)


def smoothed_probability(y_runs: np.ndarray) -> float:
    """``(k + 1) / (n + 2)`` from per-question mean success on a one-class partition."""
    y_runs = np.asarray(y_runs, dtype=np.float64)
    n = y_runs.shape[0]
    if n == 0:
        raise ValueError("empty training partition")
    k = float(y_runs.mean(axis=1).sum())
    return (k + 1.0) / (n + 2.0)


def _repeat_rows(X, runs: int):
    if sparse.issparse(X):
        return sparse.vstack([X[i] for i in range(X.shape[0]) for _ in range(runs)])
    return np.repeat(np.asarray(X), runs, axis=0)


def fit_quality_head(X, y_runs: np.ndarray, C: float, max_iter: int = 2000):
    """One L2 logistic head. Each question is repeated once per run with weight 1/2.

    ``y_runs`` has shape (n_questions, n_runs) and both runs are binary.
    A one-class partition returns the smoothed constant instead of a failed fit.
    """
    y_runs = np.asarray(y_runs, dtype=np.float64)
    n, n_runs = y_runs.shape
    if n_runs != 2:
        raise ValueError("quality heads expect exactly two generation runs")
    info = {"C": float(C), "constant": False, "converged": True, "n_iter": None}
    if np.unique(y_runs).size < 2:
        probability = smoothed_probability(y_runs)
        info["constant"] = True
        info["probability"] = probability
        return ConstantProbability(probability), info
    X_rows = _repeat_rows(X, n_runs)
    y = y_runs.reshape(-1).astype(np.int64)
    weights = np.full(y.shape, 0.5, dtype=np.float64)
    model = LogisticRegression(C=float(C), solver="lbfgs", max_iter=int(max_iter), random_state=0)
    model.fit(X_rows, y, sample_weight=weights)
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        used = int(np.max(n_iter))
        info["n_iter"] = used
        info["converged"] = used < int(max_iter)
    return model, info


def predict_quality(model, X) -> np.ndarray:
    if isinstance(model, ConstantProbability):
        return model.predict(X)
    probabilities = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    if 1 not in classes:
        raise ValueError("logistic head has no positive class")
    return np.asarray(probabilities[:, classes.index(1)], dtype=np.float64)


def fit_cost_head(X, mean_scaled_cost: np.ndarray, alpha: float):
    """Ridge on the per-question mean cost in units of ``cost_unit``. A constant target uses the mean."""
    target = np.asarray(mean_scaled_cost, dtype=np.float64)
    info = {"alpha": float(alpha), "constant": False, "solver": None}
    if target.size == 0:
        raise ValueError("empty cost training partition")
    if np.allclose(target, target[0]):
        info["constant"] = True
        info["value"] = float(target[0])
        return ConstantCost(float(target[0])), info
    solver = "lsqr" if sparse.issparse(X) else "cholesky"
    model = Ridge(alpha=float(alpha), solver=solver, random_state=0, tol=1e-8)
    model.fit(X, target)
    info["solver"] = solver
    return model, info


def question_log_loss(y_runs: np.ndarray, probability: np.ndarray, eps: float = PROB_CLIP) -> np.ndarray:
    """Per-question mean log loss across runs. ``probability`` is one number per question."""
    y_runs = np.asarray(y_runs, dtype=np.float64)
    p = clip_probability(np.asarray(probability, dtype=np.float64), eps)
    p = np.broadcast_to(p[:, None], y_runs.shape)
    losses = -(y_runs * np.log(p) + (1.0 - y_runs) * np.log(1.0 - p))
    return losses.mean(axis=1)


def question_brier(y_runs: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.broadcast_to(np.asarray(probability, dtype=np.float64)[:, None], y_runs.shape)
    return np.square(p - np.asarray(y_runs, dtype=np.float64)).mean(axis=1)


class LengthBuilder:
    def __init__(self, texts: Sequence[str]):
        self.raw = length_features(texts)
        self.scaler: Optional[StandardScaler] = None

    def fit(self, idx: np.ndarray) -> None:
        self.scaler = StandardScaler().fit(self.raw[np.asarray(idx)])

    def transform(self, idx: np.ndarray):
        if self.scaler is None:
            raise RuntimeError("length scaler was used before fit")
        return self.scaler.transform(self.raw[np.asarray(idx)])


class TfidfBuilder:
    """TF-IDF fit on the given training questions only, plus standardized length features."""

    def __init__(self, texts: Sequence[str], min_df: int = 2):
        self.texts = list(texts)
        self.min_df = int(min_df)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.length = length_features(self.texts)
        self.empty_vocabulary = False

    def fit(self, idx: np.ndarray) -> None:
        idx = np.asarray(idx)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=self.min_df, sublinear_tf=True, norm="l2")
        documents = [self.texts[int(i)] for i in idx]
        try:
            self.vectorizer.fit(documents)
            self.empty_vocabulary = False
        except ValueError:
            self.vectorizer = None
            self.empty_vocabulary = True
        self.scaler = StandardScaler().fit(self.length[idx])

    def transform(self, idx: np.ndarray):
        if self.scaler is None:
            raise RuntimeError("tf-idf builder was used before fit")
        idx = np.asarray(idx)
        length = sparse.csr_matrix(self.scaler.transform(self.length[idx]))
        if self.vectorizer is None:
            return length
        documents = [self.texts[int(i)] for i in idx]
        return sparse.hstack([self.vectorizer.transform(documents), length], format="csr")


class SemanticBuilder:
    """L2-normalized frozen embeddings plus length features standardized on the training questions."""

    def __init__(self, texts: Sequence[str], embeddings: np.ndarray):
        self.embeddings = np.asarray(embeddings, dtype=np.float64)
        self.length = length_features(texts)
        self.scaler: Optional[StandardScaler] = None
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(texts):
            raise ValueError("embeddings are not one row per question")

    def fit(self, idx: np.ndarray) -> None:
        self.scaler = StandardScaler().fit(self.length[np.asarray(idx)])

    def transform(self, idx: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("semantic builder was used before fit")
        idx = np.asarray(idx)
        return np.hstack([self.embeddings[idx], self.scaler.transform(self.length[idx])])


def make_builder(method: str, texts: Sequence[str], embeddings: Optional[np.ndarray], min_df: int = 2):
    if method == "length":
        return LengthBuilder(texts)
    if method == "tfidf":
        return TfidfBuilder(texts, min_df=min_df)
    if method == "semantic":
        if embeddings is None:
            raise ValueError("semantic features require embeddings")
        return SemanticBuilder(texts, embeddings)
    raise ValueError(f"unknown representation {method}")


def embed_from_content_ids(content_ids: Sequence[int], window_size: int, encode_short, encode_windows):
    """Short inputs call ``encode_short``. Longer inputs are windowed without dropping tokens.

    ``encode_short`` returns one pooled vector. ``encode_windows(chunks)`` returns one
    pooled vector per content-id chunk, in order, before L2 normalization.
    """
    ids = [int(token) for token in content_ids]
    if len(ids) <= window_size:
        raw = np.asarray(encode_short(), dtype=np.float64)
        return l2_normalize(raw), raw, 1, [ids]
    chunks = chunk_content_ids(ids, window_size)
    if [token for chunk in chunks for token in chunk] != ids:
        raise RuntimeError("chunking dropped or reordered content tokens")
    pooled = np.asarray(encode_windows(chunks), dtype=np.float64)
    weights = np.array([len(chunk) for chunk in chunks], dtype=np.float64)
    combined = combine_window_vectors(pooled, weights)
    return l2_normalize(combined), combined, len(chunks), chunks


def _content_ids(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def _with_special_tokens(tokenizer, content_ids: Sequence[int]) -> List[int]:
    """CLS + content + SEP, using token ids only. MiniLM adds exactly those two specials."""
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    n_special = int(tokenizer.num_special_tokens_to_add(pair=False))
    if n_special != 2 or cls_id is None or sep_id is None:
        raise RuntimeError(f"chunking expects CLS and SEP, got {n_special} specials")
    return [int(cls_id), *[int(token) for token in content_ids], int(sep_id)]


def _pooled_from_token_windows(model, chunks: Sequence[Sequence[int]]) -> np.ndarray:
    """Pool each content window with the sentence-transformer modules, without retokenizing text."""
    import torch

    tokenizer = model.tokenizer
    sequences = [_with_special_tokens(tokenizer, chunk) for chunk in chunks]
    width = max(len(seq) for seq in sequences)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    input_ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
    attention = np.zeros((len(sequences), width), dtype=np.int64)
    for row, seq in enumerate(sequences):
        input_ids[row, :len(seq)] = seq
        attention[row, :len(seq)] = 1
    features = {
        "input_ids": torch.tensor(input_ids, device=model.device),
        "attention_mask": torch.tensor(attention, device=model.device),
    }
    probe = model.tokenize(["."])
    if "token_type_ids" in probe:
        features["token_type_ids"] = torch.zeros_like(features["input_ids"])
    with torch.no_grad():
        output = model(features)
    return output["sentence_embedding"].detach().cpu().numpy()


def encode_texts(model, texts: Sequence[str]) -> dict:
    """Frozen evaluation encoding. Short questions match ``model.encode`` before the final L2."""
    tokenizer = model.tokenizer
    specials = int(tokenizer.num_special_tokens_to_add(pair=False))
    window_size = int(model.max_seq_length) - specials
    if window_size < 1:
        raise RuntimeError("model sequence limit leaves no room for content tokens")
    vectors = []
    raws = []
    token_counts = []
    window_counts = []
    errors = []
    for index, text in enumerate(texts):
        try:
            content = _content_ids(tokenizer, text)
            def encode_short(text=text):
                return model.encode([text], normalize_embeddings=False, convert_to_numpy=True,
                                     show_progress_bar=False)[0]

            def encode_windows(chunks, model=model):
                return _pooled_from_token_windows(model, chunks)

            final, raw, n_windows, chunks = embed_from_content_ids(
                content, window_size, encode_short, encode_windows)
            if [token for chunk in chunks for token in chunk] != content:
                raise RuntimeError("content tokens were not all retained")
            vectors.append(final)
            raws.append(np.asarray(raw, dtype=np.float64))
            token_counts.append(len(content))
            window_counts.append(n_windows)
        except Exception as exc:
            errors.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            raise RuntimeError(f"tokenizer or encoder failed on text {index}: {exc}") from exc
    stacked = np.stack(vectors).astype(np.float32)
    return {
        "embeddings": stacked,
        "token_counts": token_counts,
        "window_counts": window_counts,
        "errors": errors,
        "window_size": window_size,
        "max_seq_length": int(model.max_seq_length),
        "n_long": int(sum(count > 1 for count in window_counts)),
    }


def load_sentence_transformer(encoder: str, revision: str, device: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder, revision=revision, device=device)
    model.eval()
    return model


def embedding_device() -> Tuple[str, str]:
    """Prefer CUDA when it is already present. Do not use MPS; its kernels are not pinned here."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda", name
    return "cpu", "cpu"
