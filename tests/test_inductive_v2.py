"""Pending execution: the user requested approval before any new model run."""
from array import array
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mtg_gan.data import domain_dir, prepare_domain
from mtg_gan.dataset import dataset_for, collate_graphs
from mtg_gan.factory import make_model
from mtg_gan.losses import balanced_edge_loss, normalized_spectral_loss
from mtg_gan.scoring import fit_component_normalizer, combine_scores


@pytest.fixture
def v2_config(tiny_config, tmp_path):
    cfg = tiny_config
    cfg.model.architecture = "inductive_v2"
    cfg.data.cache_root = str(tmp_path/"prepared")
    cfg.data.max_nodes = 1000
    cfg.data.candidate_nodes = 8
    cfg.train.source_checkpoint_dir = str(tmp_path/"source")
    cfg.train.reconstruction_weight = 1.
    cfg.train.spectral_trace = 0.
    cfg.train.warmup_epochs = 1
    cfg.train.source_validation = True
    cfg.score.component_normalization = "robust"
    cfg.score.dynamic = False
    cfg.score.novelty_weight = .25
    cfg.validate()
    return cfg


def prepare_features(cfg):
    # Test-only embeddings, never a production fallback.
    for role in ("source", "target"):
        prepare_domain(cfg, role)
        folder = domain_dir(cfg, role)
        vocab = json.loads((folder/"vocabulary.json").read_text(encoding="utf-8"))
        features = np.random.default_rng(41).normal(size=(len(vocab), cfg.model.input_dim)).astype(np.float32)
        np.save(folder/"features.npy", features)
        (folder/"semantics.json").write_text(json.dumps({"test_fixture": True}))


def test_v2_permutation_and_padding(v2_config):
    cfg = v2_config
    model = make_model(cfg).eval()
    torch.manual_seed(42)
    n, w = 6, cfg.data.context_graphs
    features = torch.randn(1, n, cfg.model.input_dim)
    adjacency = torch.zeros(1, w, n, n)
    adjacency[:, :, 0, 1] = 1
    adjacency[:, :, 1, 2] = 1
    valid = torch.ones(1, n, dtype=torch.bool)
    present = torch.zeros(1, w, n, dtype=torch.bool)
    present[:, :, :3] = True
    permutation = torch.tensor([4, 2, 0, 5, 1, 3])
    with torch.no_grad():
        first = model.generator(features, adjacency, valid, present=present, stochastic=False)["prediction"]
        second = model.generator(features[:, permutation], adjacency[:, :, permutation][:, :, :, permutation],
                                 valid[:, permutation], present=present[:, :, permutation], stochastic=False)["prediction"]
        torch.testing.assert_close(second, first[:, permutation][:, :, permutation], atol=1e-5, rtol=1e-5)
        one = model.discriminator(adjacency[:, -1], features, valid, present=present[:, -1])
        two = model.discriminator(adjacency[:, -1, permutation][:, :, permutation], features[:, permutation],
                                   valid[:, permutation], present=present[:, -1, permutation])
        torch.testing.assert_close(one, two, atol=1e-5, rtol=1e-5)
        padded = model.generator(F.pad(features, (0, 0, 0, 2)), F.pad(adjacency, (0, 2, 0, 2)),
                                  F.pad(valid, (0, 2)), present=F.pad(present, (0, 2)), stochastic=False)["prediction"]
        torch.testing.assert_close(first, padded[:, :n, :n])
        assert (padded[:, n:] == 0).all()


def test_v2_parameters_do_not_depend_on_vocabulary_size(v2_config):
    first = make_model(v2_config)
    v2_config.data.max_nodes = 50000
    v2_config.data.candidate_nodes = 512
    second = make_model(v2_config)
    assert sum(p.numel() for p in first.parameters()) == sum(p.numel() for p in second.parameters())
    assert first.generator.query.weight.shape == second.generator.query.weight.shape


def test_future_observation_does_not_change_generator_inputs(v2_config):
    cfg = v2_config
    prepare_features(cfg)
    ds = dataset_for(cfg, "target", "train")
    first = ds[0]
    assert "label" not in first and ds.labels == {}
    graph_id = first["graph_id"]
    assert 0 not in first["global_node_ids"].tolist()  # Reserved UNKNOWN was never observed in training.
    ds.close()
    with sqlite3.connect(domain_dir(cfg, "target")/"graphs.sqlite") as conn:
        conn.execute("UPDATE graphs SET events=?,edges='[]',event_count=1 WHERE id=?",
                     (zlib.compress(array("I", [0]).tobytes()), graph_id))
    changed = dataset_for(cfg, "target", "train")
    second = changed[0]
    for key in ("features", "adjacency", "valid", "node_present", "global_node_ids"):
        torch.testing.assert_close(first[key], second[key])
    assert second["novelty"] == 1
    changed.close()


def test_v2_segment_boundaries(v2_config, tmp_path):
    cfg = v2_config
    manifest = tmp_path/"sampling.json"
    manifest.write_text(json.dumps({"domains": {"bgl": {"sample": cfg.source.log_path,
                                                        "segments": [{"lines": 45}, {"lines": 45}]}}}))
    cfg.source.sampling_manifest = cfg.target.sampling_manifest = str(manifest)
    prepare_features(cfg)
    ds = dataset_for(cfg, "source", "train")
    assert len(ds) == 14
    with sqlite3.connect(domain_dir(cfg, "source")/"graphs.sqlite") as conn:
        for ids in ds.examples:
            segments = {conn.execute("SELECT segment FROM graphs WHERE id=?", (i,)).fetchone()[0] for i in ids}
            assert len(segments) == 1
    ds.close()


def test_v2_losses_have_finite_gradients():
    real = torch.zeros(1, 5, 5)
    real[0, 0, 1] = real[0, 1, 2] = 1
    predicted = torch.rand(1, 5, 5, requires_grad=True)
    valid = torch.ones(1, 5, dtype=torch.bool)
    loss = normalized_spectral_loss(real, predicted, valid, 1., 0.)
    loss.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.norm() > 0
    logits = torch.zeros(1, 5, 5, requires_grad=True)
    balanced_edge_loss(logits, torch.zeros_like(real), valid).backward()
    assert torch.isfinite(logits.grad).all()


def test_v2_component_calibration(v2_config):
    channels = {"local": np.array([1., 2., 3.]), "global": np.array([100., 200., 300.]),
                "novelty": np.zeros(3)}
    normalizer = fit_component_normalizer(channels, v2_config)
    assert normalizer["local"]["center"] == 2
    assert normalizer["global"]["center"] == 200
    assert normalizer["novelty"]["scale"] == 1
    assert np.isfinite(combine_scores(channels, v2_config, normalizer)).all()


def test_v2_shared_cache_and_source_checkpoint(v2_config, tmp_path):
    from mtg_gan.config import DomainConfig
    from mtg_gan.training import pretrain, load_checkpoint

    cfg = v2_config
    prepare_features(cfg)
    assert domain_dir(cfg, "source") == domain_dir(cfg, "target")
    pretrain(cfg)
    cfg.target = DomainConfig("spirit", str(tmp_path/"target_not_prepared.log"))
    restored = load_checkpoint(cfg, "pretrain")
    assert restored.inductive


def test_v2_full_synthetic_pipeline(v2_config):
    from mtg_gan.training import pretrain, train_gan, calibrate, evaluate

    cfg = v2_config
    prepare_features(cfg)
    pretrain(cfg)
    train_gan(cfg)
    train_gan(cfg, target=True)
    calibration = calibrate(cfg)
    assert calibration["component_normalizer"] is not None
    report = evaluate(cfg)
    assert report["count"] > 0 and 0 <= report["predicted_positive_rate"] <= 1


def test_old_legacy_checkpoint_config_defaults(prepared):
    from mtg_gan.training import save_checkpoint, checkpoint_path, load_checkpoint

    cfg = prepared
    save_checkpoint(cfg, make_model(cfg), "source", [])
    path = checkpoint_path(cfg, "source")
    payload = torch.load(path, weights_only=True)
    payload["config"]["model"].pop("architecture")
    for key in ("cache_root", "candidate_nodes", "max_context_gap_seconds"):
        payload["config"]["data"].pop(key)
    torch.save(payload, path)
    restored = load_checkpoint(cfg, "source")
    assert not hasattr(restored, "inductive")
