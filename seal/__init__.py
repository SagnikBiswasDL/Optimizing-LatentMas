"""Clean-room SEAL (Steerable rEAsoning caLibration) for LatentMAS.

Reference: "SEAL: Steerable Reasoning Calibration of Large Language Models for
Free", Chen et al., arXiv:2504.07986.

This package implements a training-free steering-vector intervention whose
primary objective here is *token efficiency*: suppress redundant reflection /
transition thoughts during the LatentMAS Judger's text decoding so it produces
shorter reasoning traces while retaining accuracy.

Components:
  - thought_classifier: heuristic execution / reflection / transition labeler
  - vector_generation:  build the steering vector from labeled hidden states
  - extraction:         offline pipeline that produces the steering vector
  - hooks:              on-the-fly residual-stream intervention during decoding
"""

from .thought_classifier import classify_step, split_into_steps, THOUGHT_TYPES
from .vector_generation import build_steering_vector
from .hooks import SealSteerer
from .ces_steerer import TrainableSteerer, STEER_PHASES
from .ces import (
    length_normalized_nll_from_logits,
    ces_rank_loss,
    kl_tokenwise,
    hinge_kl_penalty,
    combined_ces_objective,
)
from .capture import (
    ActivationRecorder,
    build_contrastive_vector,
    correctness_probe_auc,
)
from .kv_steer import KVCacheSteerer, iter_layer_kv, cache_seq_length
from .delta_bridge import (
    CoefPredictor,
    ResidualInjector,
    LayerRecorder,
    fit_pca,
    project_residual,
    oracle_verdict,
)

__all__ = [
    "classify_step",
    "split_into_steps",
    "THOUGHT_TYPES",
    "build_steering_vector",
    "SealSteerer",
    "TrainableSteerer",
    "STEER_PHASES",
    "length_normalized_nll_from_logits",
    "ces_rank_loss",
    "kl_tokenwise",
    "hinge_kl_penalty",
    "combined_ces_objective",
    "ActivationRecorder",
    "build_contrastive_vector",
    "correctness_probe_auc",
    "KVCacheSteerer",
    "iter_layer_kv",
    "cache_seq_length",
    "CoefPredictor",
    "ResidualInjector",
    "LayerRecorder",
    "fit_pca",
    "project_residual",
    "oracle_verdict",
]
