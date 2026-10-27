"""Sparse SQLite records become dense tensors only for the current mini-batch."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config, DataConfig, DomainConfig
from .data import domain_dir, graph_edges, read_graph_db


class GraphDataset(Dataset):
    def __init__(self, cfg: Config, role: str, split: str, *, normal_only: bool = False,
                 include_labels: bool = False):
        if role == "target" and normal_only:
            raise ValueError("Target labels cannot select training or calibration examples")
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        self.cfg, self.role, self.split = cfg, role, split
        self.include_labels = include_labels
        self.folder = domain_dir(cfg, role)
        self.features = np.load(self.folder / "features.npy").astype(np.float32)
        vocab = json.loads((self.folder / "vocabulary.json").read_text(encoding="utf-8"))
        if self.features.shape != (cfg.data.max_nodes, cfg.model.input_dim):
            raise ValueError("Cached feature shape disagrees with current config; re-encode")
        prepared = json.loads((self.folder / "preparation.json").read_text(encoding="utf-8"))
        prepared_settings = {k: v for k, v in vars(DataConfig(**prepared["data"])).items() if k != "max_nodes"}
        current_settings = {k: v for k, v in vars(cfg.data).items() if k != "max_nodes"}
        # max_nodes is a preparation capacity ceiling, not part of sparse edges.
        # Padding can be resized before encoding without changing template IDs.
        if len(vocab) > cfg.data.max_nodes:
            raise ValueError("Current node capacity is smaller than the prepared vocabulary")
        if (prepared_settings != current_settings or vars(DomainConfig(**prepared["domain"])) != vars(getattr(cfg, role))
                or prepared["seed"] != cfg.train.seed):
            raise ValueError("Preparation settings changed; select the original config or a fresh output_dir")
        self.valid = np.arange(cfg.data.max_nodes) < len(vocab)
        self.hdfs = getattr(cfg, role).name.lower() == "hdfs"
        self.connection = None
        # Read labels for source selection or explicit evaluation only.
        label_column = ",label" if normal_only or include_labels else ""
        with read_graph_db(self.folder / "graphs.sqlite") as conn:
            session_filter = " AND json_array_length(events)>=2" if self.hdfs else ""
            rows = conn.execute(f"SELECT id{label_column} FROM graphs WHERE split=?{session_filter} ORDER BY start,id", (split,)).fetchall()
        self.ids = [row[0] for row in rows]
        self.labels = {row[0]: row[1] for row in rows} if label_column else {}
        if self.hdfs:
            self.examples = [(i,) for i in self.ids if not normal_only or self.labels[i] == 0]
        else:
            w = cfg.data.context_graphs
            self.examples = [tuple(self.ids[i-w:i+1]) for i in range(w, len(self.ids))
                             if not normal_only or all(self.labels[j] == 0 for j in self.ids[i-w:i+1])]

    def __len__(self) -> int:
        return len(self.examples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["connection"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        if self.connection is None:
            self.connection = read_graph_db(self.folder / "graphs.sqlite")
        records = [self.connection.execute("SELECT edges,events FROM graphs WHERE id=?", (i,)).fetchone()
                   for i in self.examples[index]]
        n, w = self.cfg.data.max_nodes, self.cfg.data.context_graphs
        target_ids = json.loads(records[-1][1])
        if self.hdfs:
            # A session is independent; predict its last portion from earlier prefixes.
            length = len(target_ids)
            ends = [max(1, min(length - 1, int(length * (i + 1) / (w + 1)))) for i in range(w)]
            edge_lists = [graph_edges(target_ids[:end], self.cfg.data.order) for end in ends]
            edge_lists.append(json.loads(records[-1][0]))
        else:
            edge_lists = [json.loads(record[0]) for record in records]
        graphs = []
        accumulated = np.zeros((n, n), dtype=np.float32)
        for edges in edge_lists:
            current = np.zeros_like(accumulated)
            if edges:
                array = np.asarray(edges, dtype=np.int64)
                current[array[:, 0], array[:, 1]] = 1.0
            if self.cfg.data.cumulative:
                accumulated = np.maximum(accumulated, current)
                current = accumulated.copy()
            graphs.append(current)
        result = {"adjacency": torch.from_numpy(np.stack(graphs[:-1])),
                  "features": torch.from_numpy(self.features),
                  "valid": torch.from_numpy(self.valid),
                  "target": torch.from_numpy(graphs[-1]),
                  "event_ids": torch.tensor(target_ids, dtype=torch.long),
                  "graph_id": self.examples[index][-1]}
        if self.include_labels:
            result["label"] = self.labels[self.examples[index][-1]]
        return result

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def collate_graphs(items: list[dict]) -> dict:
    if "node_present" in items[0]:
        from .inductive_dataset import collate_inductive
        return collate_inductive(items)
    result = {key: torch.stack([item[key] for item in items])
              for key in ("adjacency", "features", "valid", "target")}
    result["event_ids"] = [item["event_ids"] for item in items]
    result["graph_id"] = [item["graph_id"] for item in items]
    if "label" in items[0]:
        result["label"] = torch.tensor([item["label"] for item in items])
    return result


def dataset_for(cfg, role, split, **kwargs):
    if cfg.model.architecture == "inductive_v2":
        from .inductive_dataset import InductiveDataset
        return InductiveDataset(cfg, role, split, **kwargs)
    return GraphDataset(cfg, role, split, **kwargs)
