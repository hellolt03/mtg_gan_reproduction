"""Separated pretraining/source/adaptation/calibration/evaluation entry points."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, DataConfig, ModelConfig, ScoreConfig
from .data import domain_dir
from .dataset import dataset_for, collate_graphs
from .losses import (gradient_penalty, info_nce, perturbation_loss, spectral_loss,
                     normalized_spectral_loss, balanced_edge_loss)
from .model import pair_mask
from .factory import make_model, generate, discriminate
from .scoring import CausalThreshold, anomaly_scores, metrics, fit_component_normalizer, combine_scores


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(cfg: Config):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.train.device == "auto" else torch.device(cfg.train.device)


def move(batch: dict, device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@contextmanager
def loader_for(cfg, role, split, *, normal_only=False, include_labels=False, shuffle=False):
    dataset = dataset_for(cfg, role, split, normal_only=normal_only, include_labels=include_labels)
    try:
        if not len(dataset):
            raise ValueError(f"No usable {role}/{split} examples; need more sequences/context or normal data")
        loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=shuffle,
                            collate_fn=collate_graphs, num_workers=0)
        yield loader
    finally:
        dataset.close()


def preparation_digest(cfg: Config, stage=None) -> str:
    digest = hashlib.sha256()
    roles = ("source",) if cfg.model.architecture == "inductive_v2" and stage in {"source", "pretrain"} else ("source", "target")
    names = ("preparation.json", "vocabulary.json", "features.npy")
    if cfg.model.architecture == "inductive_v2":
        names += ("training_template_ids.json", "semantics.json")
    for role in roles:
        for name in names:
            path = domain_dir(cfg, role) / name
            if not path.is_file():
                raise FileNotFoundError(f"Prepare and encode both domains first: {path}")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(cfg, stage):
    folder = cfg.train.source_checkpoint_dir if stage in {"pretrain", "source"} and cfg.train.source_checkpoint_dir else cfg.output_dir
    return Path(folder)/f"{stage}.pt"


def source_training_signature(cfg):
    return {k: v for k, v in vars(cfg.train).items() if k not in {"target_epochs", "device", "source_checkpoint_dir"}}


def save_checkpoint(cfg, model, stage, history, selection=None):
    folder = checkpoint_path(cfg, stage).parent
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": cfg.to_dict(), "stage": stage,
               "history": history, "preparation_digest": preparation_digest(cfg, stage),
               "source_training_signature": source_training_signature(cfg), "selection": selection}
    torch.save(payload, checkpoint_path(cfg, stage))
    (folder / f"{stage}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (folder / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(cfg: Config, stage: str):
    path = checkpoint_path(cfg, stage)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if vars(ModelConfig(**checkpoint["config"]["model"])) != vars(cfg.model) or vars(DataConfig(**checkpoint["config"]["data"])) != vars(cfg.data):
        raise ValueError("Checkpoint architecture/data settings differ from config")
    if checkpoint["preparation_digest"] != preparation_digest(cfg, stage):
        raise ValueError("Prepared templates/features changed since this checkpoint was written")
    if checkpoint["config"]["train"]["seed"] != cfg.train.seed:
        raise ValueError("Checkpoint seed differs from the requested seed")
    if cfg.model.architecture == "inductive_v2":
        if checkpoint.get("source_training_signature") != source_training_signature(cfg):
            raise ValueError("Source training settings changed; use a new source checkpoint directory")
        if stage == "target" and checkpoint["config"]["train"]["target_epochs"] != cfg.train.target_epochs:
            raise ValueError("Target training settings changed; use a new transfer directory")
    model = make_model(cfg).to(device_for(cfg))
    model.load_state_dict(checkpoint["model"])
    return model


def pretrain(cfg: Config) -> list:
    if cfg.train.pretrain_epochs <= 0:
        raise ValueError("Set positive pretrain_epochs or use train-source --skip-pretrain")
    seed_everything(cfg.train.seed)
    device = device_for(cfg)
    model = make_model(cfg).to(device)
    encoder = model.generator.encoder
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.train.learning_rate)
    history = []
    with loader_for(cfg, "source", "train", normal_only=True, shuffle=True) as loader:
        for epoch in range(cfg.train.pretrain_epochs):
            model.train()
            total, count, informative = 0., 0, 0
            for batch in loader:
                batch = move(batch, device)
                adjacency = batch["adjacency"][:, -1]
                features, valid = batch["features"], batch["valid"]
                active = batch["node_present"][:, -1] if "node_present" in batch else valid & ((adjacency.sum(-1) + adjacency.sum(-2)) > 0)
                if active.sum() < 2:
                    continue
                embeddings = []
                for _ in range(2):
                    a = adjacency * (torch.rand_like(adjacency) >= cfg.train.augment_drop)
                    x = features * (torch.rand_like(features) >= cfg.train.augment_drop)
                    embeddings.append(encoder(x, a, active if "node_present" in batch else valid)[active])
                # Bound quadratic node-negative memory, without selecting by target labels.
                indices = torch.randperm(len(embeddings[0]), device=device)[:512]
                loss = info_nce(embeddings[0][indices], embeddings[1][indices], cfg.train.temperature)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                count += 1
                informative += len(indices)
            if not count:
                raise ValueError("GCL found no graphs with at least two active node examples")
            row = {"epoch": epoch + 1, "info_nce": total/count, "node_pairs": informative}
            history.append(row)
            print(json.dumps(row), flush=True)
    save_checkpoint(cfg, model, "pretrain", history)
    return history


def adversarial_step(model, batch, cfg, generator_optimizer, critic_optimizer, progress, probe=False):
    """One G step and configured critic steps; target batches have NO label key."""
    g, d = model.generator, model.discriminator
    for parameter in d.parameters():
        parameter.requires_grad_(True)
    for _ in range(cfg.train.critic_steps):
        with torch.no_grad():
            fake = generate(model, batch, progress=progress)["prediction"]
        # Paper convention: minimize D(real)-D(fake); lower critic means normal.
        wasserstein = discriminate(model, batch, batch["target"], observed=True).mean() - discriminate(model, batch, fake).mean()
        penalty = gradient_penalty(lambda value: discriminate(model, batch, value), batch["target"], fake, batch["valid"])
        critic_loss = wasserstein + cfg.train.gradient_penalty * penalty
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optimizer.step()
    for parameter in d.parameters():
        parameter.requires_grad_(False)
    output = generate(model, batch, progress=progress)
    adversarial = discriminate(model, batch, output["prediction"]).mean()
    contrast = perturbation_loss(output["original_features"], output["masked_features"],
                                cfg.train.contrast_margin, cfg.train.contrast_mode)
    spectral_function = normalized_spectral_loss if hasattr(model, "inductive") else spectral_loss
    spectrum = spectral_function(batch["target"], output["prediction"], batch["valid"],
                             cfg.train.heat_time, cfg.train.spectral_trace) if cfg.train.spectral_weight else adversarial * 0
    valid_pairs = pair_mask(batch["valid"])
    reconstruction = (balanced_edge_loss(output["logits"], batch["target"], batch["valid"])
                      if hasattr(model, "inductive") else
                      (batch["target"] - output["prediction"]).square()[valid_pairs].mean())
    budget = output.get("mask_budget", adversarial*0)
    loss = (adversarial + cfg.train.contrast_weight * contrast + cfg.train.spectral_weight * spectrum
            + cfg.train.reconstruction_weight * reconstruction + cfg.train.mask_budget_weight * budget)
    probe_value = None
    if probe and cfg.train.spectral_weight:
        probe_value = torch.autograd.grad(spectrum, output["prediction"], retain_graph=True)[0].norm()
    generator_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    generator_optimizer.step()
    for parameter in d.parameters():
        parameter.requires_grad_(True)
    result = {"generator": float(loss.detach()), "critic": float(critic_loss.detach()),
            "gradient_penalty": float(penalty.detach()), "contrast": float(contrast.detach()),
            "spectral": float(spectrum.detach()), "reconstruction": float(reconstruction.detach())}
    if hasattr(model, "inductive"):
        result.update(mask_budget=float(budget.detach()),
                      predicted_density=float(output["prediction"][valid_pairs].mean().detach()),
                      observed_density=float(batch["target"][valid_pairs].mean()))
        if probe_value is not None:
            result["spectral_gradient_norm"] = float(probe_value.detach())
    return result


def train_gan(cfg: Config, *, target: bool = False, skip_pretrain: bool = False):
    seed_everything(cfg.train.seed)
    role, stage = ("target", "target") if target else ("source", "source")
    epochs = cfg.train.target_epochs if target else cfg.train.source_epochs
    if epochs <= 0:
        raise ValueError(f"{role}_epochs must be positive")
    model = load_checkpoint(cfg, "source" if target else "pretrain") if target or not skip_pretrain else make_model(cfg).to(device_for(cfg))
    for parameter in model.generator.encoder.parameters():
        parameter.requires_grad_(not cfg.train.freeze_encoder)
    generator_optimizer = torch.optim.Adam((p for p in model.generator.parameters() if p.requires_grad),
                                            lr=cfg.train.learning_rate, betas=(.5, .9))
    critic_lr = cfg.train.critic_learning_rate if hasattr(model, "inductive") else cfg.train.learning_rate
    critic_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=critic_lr, betas=(.5, .9))
    history = []
    best_state, best_value, best_epoch, stale = None, float("inf"), None, 0
    with loader_for(cfg, role, "train", normal_only=not target, shuffle=True) as loader:
        if hasattr(model, "inductive") and not target:
            for epoch in range(cfg.train.warmup_epochs):
                model.train()
                total = 0.
                for batch in loader:
                    batch = move(batch, device_for(cfg))
                    output = generate(model, batch, progress=0.)
                    loss = cfg.train.reconstruction_weight*balanced_edge_loss(output["logits"], batch["target"], batch["valid"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite prediction warmup")
                    generator_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    generator_optimizer.step()
                    total += float(loss.detach())
                row = {"phase": "prediction_warmup", "epoch": epoch+1, "reconstruction": total/len(loader)}
                history.append(row)
                print(json.dumps(row), flush=True)
        for epoch in range(epochs):
            model.train()
            totals, counts = {}, {}
            for index, batch in enumerate(loader):
                batch = move(batch, device_for(cfg))
                if "label" in batch:
                    raise RuntimeError("Training batches must not expose labels")
                progress = (epoch * len(loader) + index) / max(1, epochs * len(loader) - 1)
                probe = hasattr(model, "inductive") and index % cfg.train.spectral_gradient_probe_interval == 0
                values = adversarial_step(model, batch, cfg, generator_optimizer, critic_optimizer, progress, probe)
                if not all(np.isfinite(v) for v in values.values()):
                    raise FloatingPointError(f"Non-finite training loss: {values}")
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.) + value
                    counts[key] = counts.get(key, 0)+1
            row = {"epoch": epoch+1, **{k: v/counts[k] for k, v in totals.items()}}
            if hasattr(model, "inductive") and not target and cfg.train.source_validation:
                value = source_validation_loss(cfg, model)
                row["source_normal_validation_edge_loss"] = value
                if value < best_value:
                    best_value, best_epoch, stale = value, epoch+1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
            history.append(row)
            print(json.dumps(row), flush=True)
            if best_state is not None and stale >= cfg.train.early_stopping_patience:
                break
    selection = {"criterion": "fixed_final_epoch", "target_labels_used": False}
    if best_state is not None:
        model.load_state_dict(best_state)
        selection = {"criterion": "source_normal_validation_balanced_edge_loss", "epoch": best_epoch,
                     "value": best_value, "target_labels_used": False}
    save_checkpoint(cfg, model, stage, history, selection)
    return history


@torch.no_grad()
def source_validation_loss(cfg, model):
    model.eval()
    total, count = 0., 0
    with loader_for(cfg, "source", "validation", normal_only=True) as loader:
        for batch in loader:
            batch = move(batch, device_for(cfg))
            output = generate(model, batch, stochastic=False)
            loss = balanced_edge_loss(output["logits"], batch["target"], batch["valid"])
            size = len(batch["graph_id"])
            total += float(loss)*size
            count += size
    value = total/count
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite source validation loss")
    return value


def checkpoint_hash(cfg: Config, stage: str):
    digest = hashlib.sha256()
    with checkpoint_path(cfg, stage).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate(cfg: Config, stage: str = "target") -> dict:
    model = load_checkpoint(cfg, stage).eval()
    target_policy = cfg.score.calibration == "target_unlabeled_train"
    role, split = ("target", "train") if target_policy else ("source", "validation")
    channels = {key: [] for key in ("local", "global", "novelty")}
    with loader_for(cfg, role, split, normal_only=not target_policy) as loader:
        for batch in loader:
            output = anomaly_scores(model, move(batch, device_for(cfg)), cfg)
            for key in channels:
                channels[key].extend(output[key].cpu().tolist())
    normalizer = fit_component_normalizer(channels, cfg)
    scores = combine_scores({key: np.asarray(value) for key, value in channels.items()}, cfg, normalizer)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Calibration returned non-finite scores")
    result = {"threshold": float(np.quantile(scores, cfg.score.quantile)), "count": len(scores),
              "policy": cfg.score.calibration, "score_config": vars(cfg.score), "stage": stage,
              "component_normalizer": normalizer,
              "checkpoint_sha256": checkpoint_hash(cfg, stage)}
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / f"{stage}_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate(cfg: Config, role: str = "target", stage: str = "target", split: str = "test") -> dict:
    if split == "train":
        raise ValueError("Evaluation must use validation or test")
    model = load_checkpoint(cfg, stage).eval()
    calibration = json.loads((Path(cfg.output_dir) / f"{stage}_calibration.json").read_text(encoding="utf-8"))
    if calibration["checkpoint_sha256"] != checkpoint_hash(cfg, stage) or vars(ScoreConfig(**calibration["score_config"])) != vars(cfg.score):
        raise ValueError("Checkpoint/scoring settings changed after calibration; calibrate again")
    normalizer = calibration.get("component_normalizer")
    if cfg.score.component_normalization == "robust" and normalizer is None:
        raise ValueError("Robust score calibration is missing; calibrate this checkpoint first")
    threshold = CausalThreshold(calibration["threshold"], cfg.score.quantile,
                                cfg.score.dynamic_history, cfg.score.dynamic_min_history, cfg.score.dynamic)
    labels, scores, predictions = [], [], []
    output_path = Path(cfg.output_dir) / f"{stage}_{role}_{split}_predictions.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["graph_id", "label", "score", "local", "global", "novelty",
                                                   "threshold", "prediction", "top_nodes"])
        writer.writeheader()
        with loader_for(cfg, role, split, include_labels=True) as loader:
            for batch in loader:
                output = anomaly_scores(model, move(batch, device_for(cfg)), cfg)
                output["score"] = combine_scores(output, cfg, normalizer)
                for i, score in enumerate(output["score"].cpu().tolist()):
                    prediction, used = threshold.step(score)
                    label = int(batch["label"][i])
                    labels.append(label)
                    scores.append(score)
                    predictions.append(prediction)
                    writer.writerow({"graph_id": batch["graph_id"][i], "label": label, "score": score,
                                     "local": float(output["local"][i]), "global": float(output["global"][i]),
                                     "novelty": float(output["novelty"][i]),
                                     "threshold": used, "prediction": prediction,
                                     "top_nodes": json.dumps(output["top_nodes"][i])})
    result = {**metrics(labels, scores, predictions), "role": role, "stage": stage, "split": split,
              "calibration_policy": calibration["policy"], "dynamic_threshold": cfg.score.dynamic}
    (Path(cfg.output_dir) / f"{stage}_{role}_{split}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
