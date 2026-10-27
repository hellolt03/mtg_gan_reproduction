import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from mtg_gan.config import Config, DataConfig, DomainConfig, ModelConfig, ScoreConfig, TrainConfig
from mtg_gan.data import domain_dir, prepare_domain

torch.set_num_threads(1)


@pytest.fixture
def tiny_config(tmp_path):
    raw = tmp_path / "synthetic_bgl.log"
    start = datetime(2020, 1, 1)
    lines = []
    for i in range(90):
        date = (start + timedelta(seconds=i)).strftime("%Y-%m-%d-%H.%M.%S.%f")
        message = ["service initialized", "request received", "connection closed"][i % 3]
        if i >= 72 and i % 5 == 0:
            message = "unknown catastrophic failure"
        label = "ERROR" if i >= 72 and i % 5 == 0 else "-"
        lines.append(f"{label} {1577836800+i} 2020.01.01 node {date} node RAS KERNEL INFO {message}\n")
    raw.write_text("".join(lines), encoding="utf-8")
    cfg = Config(DomainConfig("bgl", str(raw), window_seconds=3, stride_seconds=3),
                 DomainConfig("bgl", str(raw), window_seconds=3, stride_seconds=3),
                 output_dir=str(tmp_path / "run"),
                 data=DataConfig(context_graphs=2, max_nodes=8),
                 model=ModelConfig(input_dim=6, hidden_dim=8, gcn_layers=2, tcn_layers=2,
                                   dropout=0, noise_dim=2, critic_hidden=8),
                 train=TrainConfig(batch_size=2, pretrain_epochs=1, source_epochs=1,
                                   target_epochs=1, critic_steps=1),
                 score=ScoreConfig(top_k=2, local_radius=1, dynamic_min_history=2))
    cfg.validate()
    return cfg


@pytest.fixture
def prepared(tiny_config):
    cfg = tiny_config
    rng = np.random.default_rng(23)
    for role in ("source", "target"):
        prepare_domain(cfg, role)
        folder = domain_dir(cfg, role)
        vocab = json.loads((folder / "vocabulary.json").read_text(encoding="utf-8"))
        features = np.zeros((cfg.data.max_nodes, cfg.model.input_dim), dtype=np.float32)
        features[:len(vocab)] = rng.normal(size=(len(vocab), cfg.model.input_dim))
        np.save(folder / "features.npy", features)
    return cfg
