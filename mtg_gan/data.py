"""Read-only raw inputs; disk-backed sorting, Drain fitting and graph storage."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

from .config import Config, DomainConfig

BLOCK = re.compile(r"blk_-?\d+")
MASKS = [BLOCK, re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"),
         re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"),
         re.compile(r"\b0x[0-9a-fA-F]+\b"),
         re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)")]


@dataclass(frozen=True)
class Event:
    timestamp: float
    message: str
    label: int
    blocks: tuple[str, ...] = ()


def parse_line(name: str, line: str) -> Event:
    name = name.lower()
    if name == "bgl":
        p = line.strip().split(maxsplit=5)
        if len(p) != 6:
            raise ValueError("BGL requires five metadata fields and payload")
        # Repeated location may be missing or contain spaces; do not blindly use p[9:].
        found = re.search(r"(?:^|\s)(?:RAS|NULL)\s+\S+\s+"
                          r"(?:INFO|FATAL|ERROR|WARNING|SEVERE|FAILURE)(?:\s+(.*))?$",
                          p[5], re.IGNORECASE)
        if found is None:
            raise ValueError("BGL payload lacks component/level metadata")
        # Epoch is authoritative; the human-readable timestamp uses a local TZ.
        stamp = float(p[1])
        fraction = re.search(r"\.(\d+)$", p[4])
        if fraction and stamp.is_integer():
            stamp += int(fraction[1]) / 10**len(fraction[1])
        event = Event(stamp, found.group(1) or "<EMPTY>", int(p[0] != "-"))
    elif name in {"thunderbird", "spirit"}:
        p = line.strip().split(maxsplit=8)
        if len(p) != 9:
            raise ValueError("Syslog format requires eight metadata fields and payload")
        component = re.match(r"^\S+:\s*(.*)$", p[8])
        # Strip only a daemon token, not a colon inside free-form LAuS text.
        message = component.group(1).strip() if component else p[8]
        event = Event(float(p[1]), message or "<EMPTY>", int(p[0] != "-"))
    elif name == "openstack":
        p = line.strip().split(maxsplit=2)
        if len(p) != 3:
            raise ValueError("Use local OpenStack_labeled.log (label epoch payload)")
        event = Event(float(p[1]), p[2], int(p[0] != "-"))
    elif name == "hdfs":
        p = line.strip().split(maxsplit=5)
        if len(p) != 6:
            raise ValueError("HDFS requires date time pid level component content")
        stamp = datetime.strptime(p[0] + p[1], "%y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        event = Event(stamp, p[5], -1, tuple(dict.fromkeys(BLOCK.findall(p[5]))))
    else:
        raise ValueError(f"Unsupported system {name}")
    if not np.isfinite(event.timestamp):
        raise ValueError("Non-finite timestamp")
    return event


def sanitize(message: str) -> str:
    for pattern in MASKS:
        message = pattern.sub("<*>", message)
    return " ".join(message.split())


def session_split(block: str, seed: int, cfg: Config) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{block}".encode()).digest()[:8], "big") / 2**64
    if value < cfg.data.train_fraction:
        return "train"
    if value < cfg.data.train_fraction + cfg.data.validation_fraction:
        return "validation"
    return "test"


def graph_edges(ids: list[int], order: int) -> list[list[int]]:
    """Directed binary edges between events at distances 1,...,k; retain loops."""
    edges = set()
    for distance in range(1, order + 1):
        edges.update(zip(ids[:-distance], ids[distance:]))
    return [list(edge) for edge in sorted(edges)]


def read_graph_db(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def domain_dir(cfg: Config, role: str) -> Path:
    if role not in {"source", "target"}:
        raise ValueError("role must be source or target")
    if cfg.model.architecture == "inductive_v2":
        from .cache import path_for
        return path_for(cfg, role)
    return Path(cfg.output_dir) / role


def prepare_domain(cfg: Config, role: str) -> dict:
    """No model fitting; Drain learns exclusively from this domain's train split.

    Labels are stored for evaluation. Target labels never select preprocessing
    records or vocabularies. HDFS labels are needed only to attach evaluation data.
    """
    if cfg.model.architecture == "inductive_v2":
        from .cache import prepare_cached
        return prepare_cached(cfg, role)
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    domain: DomainConfig = getattr(cfg, role)
    folder = domain_dir(cfg, role)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "graphs.sqlite"
    if final.exists():
        raise FileExistsError(f"{final} exists; use a fresh output_dir to avoid mixing preparations")
    working = folder / "preparing.sqlite"
    if working.exists():
        raise FileExistsError(f"An interrupted preparation exists at {working}; use a fresh output_dir")
    labels = {}
    if domain.name.lower() == "hdfs":
        with Path(domain.label_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["Label"].lower() not in {"normal", "anomaly"}:
                    raise ValueError(f"Unknown HDFS label {row['Label']}")
                labels[row["BlockId"]] = int(row["Label"].lower() == "anomaly")
    conn = sqlite3.connect(working)
    try:
        conn.executescript("""
        CREATE TABLE events(id INTEGER PRIMARY KEY, ts REAL, message TEXT, label INTEGER,
                            split TEXT, template INTEGER);
        CREATE TABLE sessions(block TEXT PRIMARY KEY, split TEXT, label INTEGER);
        CREATE TABLE membership(block TEXT, event INTEGER);
        CREATE TABLE graphs(id INTEGER PRIMARY KEY, split TEXT, start REAL, end REAL,
                            label INTEGER, edges TEXT, events TEXT, sequence_key TEXT);
        """)
        count = 0
        with Path(domain.log_path).open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if domain.max_lines is not None and number > domain.max_lines:
                    break
                try:
                    event = parse_line(domain.name, line)
                except ValueError as exc:
                    raise ValueError(f"{domain.log_path}:{number}: {exc}") from exc
                conn.execute("INSERT INTO events VALUES (?,?,?,?,NULL,NULL)",
                             (number, event.timestamp, event.message, event.label))
                for block in event.blocks:
                    if block not in labels:
                        raise ValueError(f"HDFS block {block} missing from label CSV")
                    conn.execute("INSERT OR IGNORE INTO sessions VALUES (?,?,?)",
                                 (block, session_split(block, cfg.train.seed, cfg), labels[block]))
                    conn.execute("INSERT INTO membership VALUES (?,?)", (block, number))
                count += 1
                if number % 100000 == 0:
                    conn.commit()
                    print(f"{role}: ingested {number:,} lines", flush=True)
        if count < 3:
            raise ValueError("Need at least three events")
        conn.executescript("CREATE INDEX events_time ON events(ts,id);"
                           "CREATE INDEX member_block ON membership(block,event);"
                           "CREATE INDEX member_event ON membership(event);")
        hdfs = domain.name.lower() == "hdfs"
        if hdfs:
            training_sql = ("SELECT e.message FROM events e WHERE EXISTS "
                            "(SELECT 1 FROM membership m JOIN sessions s ON s.block=m.block "
                            "WHERE m.event=e.id AND s.split='train') ORDER BY e.ts,e.id")
        else:
            # Split raw ordered records BEFORE making overlapping time windows.
            ntrain = int(count * cfg.data.train_fraction)
            nvalid = int(count * cfg.data.validation_fraction)
            conn.execute("CREATE TEMP TABLE ordering AS SELECT id,ROW_NUMBER() OVER(ORDER BY ts,id) AS r FROM events")
            conn.execute("CREATE INDEX ordering_id ON ordering(id)")
            conn.execute("UPDATE events SET split=(SELECT CASE WHEN r<=? THEN 'train' "
                         "WHEN r<=? THEN 'validation' ELSE 'test' END FROM ordering WHERE ordering.id=events.id)",
                         (ntrain, ntrain + nvalid))
            training_sql = "SELECT message FROM events WHERE split='train' ORDER BY ts,id"
        dc = TemplateMinerConfig()
        dc.drain_depth = cfg.data.drain_depth
        dc.drain_sim_th = cfg.data.drain_similarity
        miner = TemplateMiner(config=dc)
        for (message,) in conn.execute(training_sql):
            miner.add_log_message(sanitize(message))
        # Store final cluster templates; never cache an early, mutable cluster text.
        templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
        counts = Counter()
        train_ids_sql = ("SELECT DISTINCT m.event AS id FROM membership m JOIN sessions s ON m.block=s.block "
                         "WHERE s.split='train'" if hdfs else "SELECT id FROM events WHERE split='train'")
        conn.execute("CREATE TEMP TABLE training_ids AS " + train_ids_sql)
        conn.execute("CREATE UNIQUE INDEX training_ids_idx ON training_ids(id)")
        unknown_events = 0
        for event_id, message, is_train in conn.execute(
                "SELECT e.id,e.message,t.id IS NOT NULL FROM events e LEFT JOIN training_ids t ON e.id=t.id ORDER BY e.id"):
            match = miner.match(sanitize(message), full_search_strategy="fallback")
            template_id = match.cluster_id if match else 0
            conn.execute("UPDATE events SET template=? WHERE id=?", (template_id, event_id))
            if is_train:
                counts[template_id] += 1
            unknown_events += int(template_id == 0)
        vocab = ["<UNKNOWN>"] + [templates[key] for key in sorted(
            (key for key in counts if key != 0), key=lambda key: (-counts[key], templates[key]))]
        if len(vocab) > cfg.data.max_nodes:
            raise ValueError(f"{len(vocab)} templates (including UNKNOWN) exceed max_nodes={cfg.data.max_nodes}; "
                             "increase capacity, no silent truncation is performed")
        text_to_slot = {text: i for i, text in enumerate(vocab)}
        slot_map = {key: text_to_slot.get(value, 0) for key, value in templates.items()}
        slot_map[0] = 0
        ngraphs = Counter()

        def save(split: str, key: str, rows) -> None:
            ids, target_labels, first, last = [], [], None, None
            for ts, template, label in rows:
                ids.append(slot_map.get(template, 0))
                target_labels.append(label)
                first = ts if first is None else first
                last = ts
            if not ids:
                return
            label = max(target_labels)
            if label not in {0, 1}:
                raise ValueError("Missing sequence labels")
            conn.execute("INSERT INTO graphs(split,start,end,label,edges,events,sequence_key) VALUES (?,?,?,?,?,?,?)",
                         (split, first, last, label, json.dumps(graph_edges(ids, cfg.data.order)), json.dumps(ids), key))
            ngraphs[split] += 1

        if hdfs:
            for block, split, label in conn.execute("SELECT block,split,label FROM sessions ORDER BY block"):
                rows = conn.execute("SELECT e.ts,e.template,? FROM membership m JOIN events e "
                                    "ON e.id=m.event WHERE m.block=? ORDER BY e.ts,e.id", (label, block))
                save(split, block, rows)
        else:
            conn.execute("CREATE INDEX events_split_time ON events(split,ts,id)")
            for split in ("train", "validation", "test"):
                bounds = conn.execute("SELECT MIN(ts),MAX(ts) FROM events WHERE split=?", (split,)).fetchone()
                if bounds[0] is None:
                    continue
                start = np.floor(bounds[0] / domain.stride_seconds) * domain.stride_seconds
                while start <= bounds[1]:
                    save(split, f"{start:.6f}", conn.execute(
                        "SELECT ts,template,label FROM events WHERE split=? AND ts>=? AND ts<? ORDER BY ts,id",
                        (split, float(start), float(start + domain.window_seconds))))
                    # Skip long empty gaps without looping over every possible window.
                    nxt = conn.execute("SELECT MIN(ts) FROM events WHERE split=? AND ts>=?",
                                       (split, float(start + domain.stride_seconds))).fetchone()[0]
                    if nxt is None:
                        break
                    next_start = start + domain.stride_seconds
                    earliest = np.floor((nxt - domain.window_seconds) / domain.stride_seconds + 1) * domain.stride_seconds
                    start = max(next_start, earliest)
        conn.commit()
        # Only graph records are needed after preparation; raw content stays in original input.
        conn.executescript("DROP TABLE membership; DROP TABLE sessions; DROP TABLE events;"
                           "CREATE INDEX graph_split_time ON graphs(split,start,id);")
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
        working.replace(final)
        stats = {"role": role, "system": domain.name, "raw_lines": count,
                 "templates": len(vocab), "unmatched_events": unknown_events,
                 "graphs": dict(ngraphs), "node_order": "train_frequency_then_template_text",
                 "partial_hdfs_sessions": bool(hdfs and domain.max_lines),
                 "domain": vars(domain), "data": vars(cfg.data), "seed": cfg.train.seed}
        (folder / "vocabulary.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "preparation.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        return stats
    finally:
        conn.close()


def encode_domain(cfg: Config, role: str) -> dict:
    """Frozen shared Sentence-BERT; no fitting on validation/test templates."""
    if cfg.model.architecture == "inductive_v2":
        from .cache import encode_cached
        return encode_cached(cfg, role)
    from sentence_transformers import SentenceTransformer

    folder = domain_dir(cfg, role)
    vocab = json.loads((folder / "vocabulary.json").read_text(encoding="utf-8"))
    try:
        encoder = SentenceTransformer(cfg.semantics.model, local_files_only=cfg.semantics.local_files_only)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load semantic model {cfg.semantics.model!r}. The local cache may contain "
            "configuration without weights. Provide a complete local SentenceTransformer model, "
            "or explicitly set semantics.local_files_only=false to allow downloading."
        ) from exc
    values = encoder.encode(vocab, batch_size=cfg.semantics.batch_size,
                            normalize_embeddings=True, show_progress_bar=True)
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(vocab), cfg.model.input_dim):
        raise ValueError(f"Encoder returned {values.shape}; model.input_dim={cfg.model.input_dim}")
    features = np.zeros((cfg.data.max_nodes, cfg.model.input_dim), dtype=np.float32)
    features[:len(vocab)] = values
    np.save(folder / "features.npy", features)
    meta = {"encoder": cfg.semantics.model, "dimension": cfg.model.input_dim, "templates": len(vocab)}
    (folder / "semantics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
