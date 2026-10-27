"""Spectral event attribution, local critic scores and causal thresholds."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .config import Config
from .data import graph_edges
from .losses import laplacian, normalized_laplacian
from .factory import generate, discriminate


@torch.no_grad()
def anomaly_scores(model, batch: dict, cfg: Config) -> dict:
    output = generate(model, batch, stochastic=False)
    predicted = output["prediction"]
    real = batch["target"]
    global_values = []
    impact = torch.full_like(real[:, :, 0], -torch.inf)
    for i, valid in enumerate(batch["valid"]):
        # Exclude zero padding before cubic eigendecomposition. This removes
        # zero eigenvalues only and preserves both score and valid-node impact.
        r, p = real[i][valid][:, valid], predicted[i][valid][:, valid]
        operator = normalized_laplacian if hasattr(model, "inductive") else laplacian
        eigenvalues, eigenvectors = torch.linalg.eigh(operator(r) - operator(p))
        global_values.append(eigenvalues.norm()/(max(1, len(r))**.5 if hasattr(model, "inductive") else 1))
        impact[i, valid] = (eigenvectors.square() * eigenvalues.abs().unsqueeze(-2)).sum(-1)
    global_score = torch.stack(global_values)
    local_scores, selected = [], []
    for i, ids_tensor in enumerate(batch["event_ids"]):
        ids = ids_tensor.tolist()
        k = min(cfg.score.top_k, int(batch["valid"][i].sum()))
        top = impact[i].topk(k).indices.tolist()
        selected.append([int(batch["global_node_ids"][i][j]) for j in top] if "global_node_ids" in batch else top)
        # For each important template choose its median occurrence. The paper's
        # 'improved top-k' is unspecified; bound critic calls by top_k even for
        # million-line windows. Median occurrence is an explicit reconstruction.
        positions = {node: [] for node in top}
        for p, event in enumerate(ids):
            if event in positions:
                positions[event].append(p)
        centers = [values[len(values)//2] for values in positions.values() if values]
        intervals = {(max(0, p-cfg.score.local_radius), min(len(ids), p+cfg.score.local_radius+1))
                     for p in centers}
        if not intervals:
            intervals = {(0, len(ids))}
        value = []
        # Stream intervals: do not allocate every full N*N local matrix at once.
        for start, end in sorted(intervals):
            local = torch.zeros_like(real[i])
            edges = [edge for edge in graph_edges(ids[start:end], cfg.data.order) if min(edge) >= 0]
            if edges:
                edge_index = torch.tensor(edges, device=real.device)
                local[edge_index[:, 0], edge_index[:, 1]] = 1
            single = {"features": batch["features"][i:i+1], "valid": batch["valid"][i:i+1]}
            present = torch.zeros_like(single["valid"])
            for node in ids[start:end]:
                if node >= 0:
                    present[0, node] = True
            value.append(discriminate(model, single, local.unsqueeze(0), present=present)[0])
        local_scores.append(torch.stack(value).mean())
    local_score = torch.stack(local_scores)
    novelty = batch.get("novelty", torch.zeros_like(global_score))
    result = {"local": local_score, "global": global_score, "novelty": novelty, "top_nodes": selected}
    result["score"] = combine_scores(result, cfg)
    return result


def fit_component_normalizer(channels, cfg):
    if cfg.score.component_normalization == "none":
        return None
    result = {}
    for key in ("local", "global", "novelty"):
        values = np.asarray(channels[key], dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite calibration components")
        center = float(np.median(values))
        lower, upper = np.quantile(values, [.25, .75])
        scale = float(upper-lower)
        if scale < 1e-8:
            scale = float(values.std())
        result[key] = {"center": center, "scale": scale if scale >= 1e-8 else 1.0}
    return result


def combine_scores(channels, cfg, normalizer=None):
    values = {}
    for key in ("local", "global", "novelty"):
        value = channels[key]
        if normalizer is not None:
            value = (value-normalizer[key]["center"])/normalizer[key]["scale"]
        values[key] = value
    return (cfg.score.local_weight*values["local"] + (1-cfg.score.local_weight)*values["global"]
            + cfg.score.novelty_weight*values["novelty"])


@dataclass
class CausalThreshold:
    """Historical quantile: decide first, then append only accepted observations.

    This is a reconstruction of the unspecified dynamic-threshold procedure.
    There is no access to current/future labels or a whole test-score quantile.
    """
    initial: float
    quantile: float
    capacity: int
    minimum: int
    enabled: bool = True

    def __post_init__(self):
        self.history = deque(maxlen=self.capacity)

    def step(self, score: float) -> tuple[int, float]:
        if not np.isfinite(score):
            raise ValueError("Non-finite anomaly score")
        threshold = self.initial
        if self.enabled and len(self.history) >= self.minimum:
            threshold = float(np.quantile(self.history, self.quantile))
        prediction = int(score > threshold)
        if self.enabled and not prediction:
            self.history.append(float(score))
        return prediction, threshold


def metrics(labels: list[int], scores: list[float], predictions: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"count": len(labels), "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "accuracy": float(accuracy_score(labels, predictions)),
            "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "average_precision": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
            "predicted_positive_rate": sum(predictions)/len(predictions),
            "tp": sum(y == 1 and p == 1 for y, p in zip(labels, predictions)),
            "fp": sum(y == 0 and p == 1 for y, p in zip(labels, predictions)),
            "tn": sum(y == 0 and p == 0 for y, p in zip(labels, predictions)),
            "fn": sum(y == 1 and p == 0 for y, p in zip(labels, predictions))}
