import json
import sqlite3

import numpy as np
import pytest

from mtg_gan.config import load_config
from mtg_gan.data import domain_dir, graph_edges, parse_line, prepare_domain
from mtg_gan.dataset import GraphDataset


def test_real_formats_strip_metadata():
    e = parse_line("bgl", "- 1117838570 2005.06.03 node 2005-06-03-15.42.50.363779 node RAS KERNEL INFO cache error corrected")
    assert e.timestamp == pytest.approx(1117838570.363779)
    assert e.message == "cache error corrected" and e.label == 0
    # Missing or space-containing repeated location, found in the real BGL file.
    assert parse_line("bgl", "X 1 d n 2005-06-03-15.42.50.000000 RAS KERNEL FATAL bad").message == "bad"
    assert parse_line("thunderbird", "- 1 d host Nov 10 00:00:01 src@host daemon[42]: unable to connect").message == "unable to connect"
    spirit = parse_line("spirit", "X 1 d host Jan 1 00:00:01 host LAuS error - do_command.c:175 - failure")
    assert spirit.label == 1 and spirit.message == "LAuS error - do_command.c:175 - failure"
    assert parse_line("openstack", "- 1.25 25746 INFO nova.server hello").timestamp == 1.25
    hdfs = parse_line("hdfs", "081109 203518 143 INFO dfs.DataNode: block blk_-123 blk_-123 blk_99")
    assert hdfs.blocks == ("blk_-123", "blk_99")


def test_ordered_edges_and_loops():
    assert graph_edges([1, 2, 2, 3], 1) == [[1, 2], [2, 2], [2, 3]]
    assert [1, 3] in graph_edges([1, 2, 3], 2)
    assert [3, 1] not in graph_edges([1, 2, 3], 2)


def test_prepare_train_only_vocabulary_and_disjoint_splits(prepared):
    cfg = prepared
    folder = domain_dir(cfg, "source")
    vocab = json.loads((folder / "vocabulary.json").read_text())
    assert all("catastrophic" not in text for text in vocab)
    with sqlite3.connect(folder / "graphs.sqlite") as db:
        intervals = {s: db.execute("SELECT MIN(start),MAX(end) FROM graphs WHERE split=?", (s,)).fetchone()
                     for s in ("train", "validation", "test")}
        assert intervals["train"][1] < intervals["validation"][0]
        assert intervals["validation"][1] < intervals["test"][0]
        assert db.execute("SELECT COUNT(*) FROM graphs WHERE split='test' AND events LIKE '%0%'").fetchone()[0] > 0
    with pytest.raises(FileExistsError):
        prepare_domain(cfg, "source")


def test_target_labels_absent_from_training_and_no_filter(prepared):
    cfg = prepared
    ds = GraphDataset(cfg, "target", "train")
    first = ds[0]
    assert "label" not in first and ds.labels == {}
    ds.close()
    # Mutating target labels cannot change input examples or their tensors.
    with sqlite3.connect(domain_dir(cfg, "target") / "graphs.sqlite") as db:
        db.execute("UPDATE graphs SET label=1-label")
    changed = GraphDataset(cfg, "target", "train")
    assert len(changed) == len(ds)
    np.testing.assert_array_equal(first["target"], changed[0]["target"])
    np.testing.assert_array_equal(first["adjacency"], changed[0]["adjacency"])
    changed.close()
    with pytest.raises(ValueError, match="Target labels"):
        GraphDataset(cfg, "target", "train", normal_only=True)


def test_config_relative_paths_and_unknown_keys(tiny_config, tmp_path):
    path = tmp_path / "config.json"
    raw = tiny_config.to_dict()
    raw["output_dir"] = "relative_run"
    path.write_text(json.dumps(raw))
    assert load_config(path).output_dir == str(tmp_path / "relative_run")
    raw["typo"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_padding_resize_preserves_sparse_graphs(prepared):
    cfg = prepared
    original = GraphDataset(cfg, "source", "train")
    first = original[0]
    original.close()
    cfg.data.max_nodes = 6
    for role in ("source", "target"):
        path = domain_dir(cfg, role) / "features.npy"
        np.save(path, np.load(path)[:6])
    resized = GraphDataset(cfg, "source", "train")
    np.testing.assert_array_equal(first["target"][:6, :6], resized[0]["target"])
    np.testing.assert_array_equal(first["adjacency"][:, :6, :6], resized[0]["adjacency"])
    resized.close()


def test_capacity_error_and_hdfs_truncation_guard(tiny_config):
    tiny_config.data.max_nodes = 1
    with pytest.raises(ValueError, match="exceed max_nodes"):
        prepare_domain(tiny_config, "source")
    tiny_config.source.name = "hdfs"
    tiny_config.source.label_path = "labels.csv"
    tiny_config.source.max_lines = 10
    with pytest.raises(ValueError, match="truncate"):
        tiny_config.validate()
