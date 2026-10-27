import json
import sqlite3
import sys
import types

import numpy as np
import pytest

from mtg_gan.data import domain_dir, encode_domain, prepare_domain, session_split
from mtg_gan.dataset import GraphDataset


def test_hdfs_complete_block_sessions_and_prefix_prediction(tiny_config, tmp_path):
    cfg = tiny_config
    log, labels = tmp_path / "hdfs.log", tmp_path / "labels.csv"
    blocks = [f"blk_{i}" for i in range(30)]
    lines = []
    for block in blocks:
        for j, message in enumerate(("Receiving", "Writing", "Closing", "Deleting")):
            lines.append(f"081109 2035{j:02d} 143 INFO dfs.Node: {message} block {block}\n")
    log.write_text("".join(lines))
    labels.write_text("BlockId,Label\n" + "".join(f"{b},Normal\n" for b in blocks))
    cfg.source.name, cfg.source.log_path, cfg.source.label_path = "hdfs", str(log), str(labels)
    stats = prepare_domain(cfg, "source")
    assert sum(stats["graphs"].values()) == len(blocks)
    folder = domain_dir(cfg, "source")
    with sqlite3.connect(folder / "graphs.sqlite") as db:
        for split, key, ids in db.execute("SELECT split,sequence_key,events FROM graphs"):
            assert split == session_split(key, cfg.train.seed, cfg)
            assert len(json.loads(ids)) == 4
    np.save(folder / "features.npy", np.ones((cfg.data.max_nodes, cfg.model.input_dim), dtype=np.float32))
    ds = GraphDataset(cfg, "source", "train", normal_only=True)
    item = ds[0]
    assert item["target"].sum() > item["adjacency"][-1].sum()
    assert "label" not in item
    ds.close()


def test_semantics_padding_and_dimension_checks(tiny_config, monkeypatch):
    cfg = tiny_config
    prepare_domain(cfg, "source")
    calls = []

    class FakeEncoder:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))

        def encode(self, vocab, **kwargs):
            return np.ones((len(vocab), cfg.model.input_dim), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeEncoder))
    info = encode_domain(cfg, "source")
    assert calls[0][1]["local_files_only"] is True
    features = np.load(domain_dir(cfg, "source") / "features.npy")
    assert np.all(features[info["templates"]:] == 0)

    class WrongEncoder(FakeEncoder):
        def encode(self, vocab, **kwargs):
            return np.zeros((len(vocab), 999))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=WrongEncoder))
    with pytest.raises(ValueError, match="input_dim"):
        encode_domain(cfg, "source")
