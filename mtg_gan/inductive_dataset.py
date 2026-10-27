"""Variable-node examples: candidates use history and training counts only."""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import decode_events, path_for
from .data import read_graph_db


class InductiveDataset(Dataset):
    def __init__(self, cfg, role, split, *, normal_only=False, include_labels=False):
        if role == "target" and normal_only:
            raise ValueError("Target labels cannot select examples")
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        self.cfg, self.role, self.split = cfg, role, split
        self.folder = path_for(cfg, role)
        self.features = np.load(self.folder/"features.npy", mmap_mode="r")
        self.training_ids = json.loads((self.folder/"training_template_ids.json").read_text())
        if self.features.shape[1] != cfg.model.input_dim:
            raise ValueError("Feature cache dimension mismatch")
        self.connection = None
        self.include_labels = include_labels
        label_column = ",label" if normal_only or include_labels else ""
        with read_graph_db(self.folder/"graphs.sqlite") as conn:
            rows = conn.execute(f"SELECT id,segment,window_start{label_column} FROM graphs WHERE split=? "
                                "ORDER BY window_start,id", (split,)).fetchall()
        self.labels = {r[0]: r[3] for r in rows} if label_column else {}
        times = {r[0]: r[2] for r in rows}
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[1]].append((row[0], row[2]))
        w = cfg.data.context_graphs
        gap = cfg.data.max_context_gap_seconds or 1.5*getattr(cfg, role).stride_seconds
        examples = []
        for records in grouped.values():
            for index in range(w, len(records)):
                chunk = records[index-w:index+1]
                if any(chunk[i][1]-chunk[i-1][1] > gap for i in range(1, len(chunk))):
                    continue
                ids = tuple(record[0] for record in chunk)
                if not normal_only or all(self.labels[i] == 0 for i in ids):
                    examples.append(ids)
        self.examples = sorted(examples, key=lambda ids: (times[ids[-1]], ids[-1]))

    def __len__(self):
        return len(self.examples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["connection"] = None
        return state

    def __getitem__(self, index):
        if self.connection is None:
            self.connection = read_graph_db(self.folder/"graphs.sqlite")
        ids = self.examples[index]
        records = [self.connection.execute("SELECT edges,events FROM graphs WHERE id=?", (i,)).fetchone() for i in ids]
        sequences = [decode_events(row[1]) for row in records]
        observed = set()
        for sequence in sequences[:-1]:
            observed.update(int(value) for value in np.unique(sequence))
        limit = self.cfg.data.candidate_nodes
        if len(observed) > limit:
            raise ValueError(f"History needs {len(observed)} nodes, candidate_nodes={limit}; "
                             "increase the limit or use a shorter window. No nodes were truncated.")
        candidates = sorted(observed)
        for node in self.training_ids:
            if len(candidates) >= limit:
                break
            if node not in observed:
                candidates.append(node)
        lookup = {node: i for i, node in enumerate(candidates)}
        n = len(candidates)
        if not n:
            raise ValueError("No history/candidate nodes")
        graphs, presences = [], []
        cumulative = np.zeros((n, n), dtype=np.float32)
        cumulative_presence = np.zeros(n, dtype=bool)
        for row, sequence in zip(records, sequences):
            adjacency = np.zeros_like(cumulative)
            for first, second in json.loads(row[0]):
                if first in lookup and second in lookup:
                    adjacency[lookup[first], lookup[second]] = 1
            present = np.zeros(n, dtype=bool)
            for node in np.unique(sequence):
                if int(node) in lookup:
                    present[lookup[int(node)]] = True
            if self.cfg.data.cumulative:
                cumulative = np.maximum(cumulative, adjacency)
                cumulative_presence |= present
                adjacency, present = cumulative.copy(), cumulative_presence.copy()
            graphs.append(adjacency)
            presences.append(present)
        # Future nodes not in the historical/training candidates are excluded
        # from generator inputs and explicitly retained as a novelty fraction.
        target_local = np.asarray([lookup.get(int(node), -1) for node in sequences[-1]], dtype=np.int64)
        result = {"adjacency": torch.from_numpy(np.stack(graphs[:-1])),
                  "features": torch.from_numpy(np.array(self.features[candidates], dtype=np.float32)),
                  "valid": torch.ones(n, dtype=torch.bool), "target": torch.from_numpy(graphs[-1]),
                  "node_present": torch.from_numpy(np.stack(presences[:-1])),
                  "target_present": torch.from_numpy(presences[-1]),
                  "novelty": torch.tensor(float((target_local < 0).mean()), dtype=torch.float32),
                  "event_ids": torch.from_numpy(target_local), "global_node_ids": torch.tensor(candidates),
                  "graph_id": ids[-1]}
        if self.include_labels:
            result["label"] = self.labels[ids[-1]]
        return result

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def collate_inductive(items):
    n = max(item["features"].shape[0] for item in items)
    b, w, f = len(items), items[0]["adjacency"].shape[0], items[0]["features"].shape[-1]
    result = {"adjacency": torch.zeros(b, w, n, n), "target": torch.zeros(b, n, n),
              "features": torch.zeros(b, n, f), "valid": torch.zeros(b, n, dtype=torch.bool),
              "node_present": torch.zeros(b, w, n, dtype=torch.bool),
              "target_present": torch.zeros(b, n, dtype=torch.bool),
              "novelty": torch.stack([item["novelty"] for item in items])}
    for i, item in enumerate(items):
        size = item["features"].shape[0]
        result["adjacency"][i, :, :size, :size] = item["adjacency"]
        result["target"][i, :size, :size] = item["target"]
        result["features"][i, :size] = item["features"]
        result["valid"][i, :size] = item["valid"]
        result["node_present"][i, :, :size] = item["node_present"]
        result["target_present"][i, :size] = item["target_present"]
    for key in ("event_ids", "global_node_ids", "graph_id"):
        result[key] = [item[key] for item in items]
    if "label" in items[0]:
        result["label"] = torch.tensor([item["label"] for item in items])
    return result
