"""Shared per-domain v2 cache. Raw files are streamed into disk-backed SQLite."""
from __future__ import annotations

from array import array
from collections import Counter, deque
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zlib

import numpy as np

SCHEMA = "inductive-domain-v2.1"


def cache_identity(cfg, role):
    domain = getattr(cfg, role)
    path = Path(domain.log_path)
    stat = path.stat()
    manifest = Path(cfg.semantics.model) / "download_manifest.json"
    segment_manifest = Path(domain.sampling_manifest) if domain.sampling_manifest else None
    return {"schema": SCHEMA, "domain": vars(domain), "raw_bytes": stat.st_size,
            "raw_mtime_ns": stat.st_mtime_ns,
            "sampling_sha256": hashlib.sha256(segment_manifest.read_bytes()).hexdigest() if segment_manifest else None,
            "split": [cfg.data.train_fraction, cfg.data.validation_fraction], "order": cfg.data.order,
            "drain": [cfg.data.drain_depth, cfg.data.drain_similarity],
            "encoder": cfg.semantics.model, "input_dim": cfg.model.input_dim,
            "encoder_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None}


def path_for(cfg, role):
    identity = cache_identity(cfg, role)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return Path(cfg.data.cache_root) / getattr(cfg, role).name.lower() / digest


def decode_events(blob):
    return np.frombuffer(zlib.decompress(blob), dtype="<u4")


def prepare_cached(cfg, role):
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from .data import parse_line, sanitize

    domain = getattr(cfg, role)
    folder = path_for(cfg, role)
    metadata_path = folder / "preparation.json"
    if (folder / "graphs.sqlite").exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["identity"] != cache_identity(cfg, role):
            raise ValueError("Cache identity mismatch")
        if metadata["templates"] > cfg.data.max_nodes:
            raise ValueError("Prepared vocabulary exceeds configured template capacity")
        return metadata
    folder.mkdir(parents=True, exist_ok=True)
    working = folder / "preparing.sqlite"
    if working.exists():
        raise FileExistsError(f"Interrupted preparation at {working}; inspect it before retrying")
    segment_ends = []
    if domain.sampling_manifest:
        source_manifest = json.loads(Path(domain.sampling_manifest).read_text(encoding="utf-8"))
        description = source_manifest["domains"][domain.name.lower()]
        if Path(description["sample"]).resolve() != Path(domain.log_path).resolve():
            raise ValueError("Sampling manifest describes a different input file")
        total = 0
        for segment in description["segments"]:
            total += segment["lines"]
            segment_ends.append(total)
    conn = sqlite3.connect(working)
    normalized = lru_cache(maxsize=20000)(sanitize)
    try:
        conn.executescript("""
        PRAGMA cache_size=-32768;
        PRAGMA temp_store=FILE;
        CREATE TABLE events(id INTEGER PRIMARY KEY,ts REAL,message TEXT,label INTEGER,
                            segment INTEGER,split TEXT,template INTEGER);
        CREATE TABLE graphs(id INTEGER PRIMARY KEY,split TEXT,start REAL,end REAL,
                            window_start REAL,segment INTEGER,label INTEGER,edges TEXT,
                            events BLOB,event_count INTEGER);
        """)
        batch, count, segment_index = [], 0, 0
        with Path(domain.log_path).open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if domain.max_lines is not None and number > domain.max_lines:
                    break
                if segment_ends:
                    while segment_index < len(segment_ends) and number > segment_ends[segment_index]:
                        segment_index += 1
                    if segment_index >= len(segment_ends):
                        raise ValueError("Input has more lines than its sampling manifest")
                try:
                    event = parse_line(domain.name, line)
                except ValueError as exc:
                    raise ValueError(f"{domain.log_path}:{number}: {exc}") from exc
                batch.append((number, event.timestamp, event.message, event.label, segment_index))
                count = number
                if len(batch) >= 5000:
                    conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,NULL,NULL)", batch)
                    conn.commit()
                    batch.clear()
                if number % 100000 == 0:
                    print(f"{domain.name}: ingested {number:,}", flush=True)
        if batch:
            conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,NULL,NULL)", batch)
        if count < 3 or (segment_ends and not domain.max_lines and count != segment_ends[-1]):
            raise ValueError("Insufficient input or sampling manifest line count mismatch")
        conn.executescript("CREATE INDEX events_time ON events(ts,id);"
                           "CREATE TEMP TABLE ordering AS SELECT id,ROW_NUMBER() OVER(ORDER BY ts,id) AS r FROM events;"
                           "CREATE UNIQUE INDEX ordering_id ON ordering(id);")
        train_end = int(count*cfg.data.train_fraction)
        valid_end = train_end + int(count*cfg.data.validation_fraction)
        conn.execute("UPDATE events SET split=(SELECT CASE WHEN r<=? THEN 'train' WHEN r<=? "
                     "THEN 'validation' ELSE 'test' END FROM ordering WHERE ordering.id=events.id)", (train_end, valid_end))
        dc = TemplateMinerConfig()
        dc.drain_depth, dc.drain_sim_th = cfg.data.drain_depth, cfg.data.drain_similarity
        miner = TemplateMiner(config=dc)
        for (message,) in conn.execute("SELECT message FROM events WHERE split='train' ORDER BY ts,id"):
            miner.add_log_message(normalized(message))

        @lru_cache(maxsize=50000)
        def match_text(text):
            cluster = miner.match(text, full_search_strategy="fallback")
            # A frozen fallback retains unseen text for inference; no refitting.
            return (cluster.get_template(), True) if cluster else (text, False)

        vocabulary, text_ids, training_counts = ["<UNKNOWN>"], {"<UNKNOWN>": 0}, Counter()
        unmatched = Counter()
        updates = []
        for event_id, message, split in conn.execute("SELECT id,message,split FROM events ORDER BY id"):
            text, known = match_text(normalized(message))
            if text not in text_ids:
                if len(vocabulary) >= cfg.data.max_nodes:
                    raise ValueError("Template capacity exceeded; increase data.max_nodes. No truncation performed.")
                text_ids[text] = len(vocabulary)
                vocabulary.append(text)
            slot = text_ids[text]
            if split == "train":
                training_counts[slot] += 1
            unmatched[split] += int(not known)
            updates.append((slot, event_id))
            if len(updates) >= 5000:
                conn.executemany("UPDATE events SET template=? WHERE id=?", updates)
                updates.clear()
        if updates:
            conn.executemany("UPDATE events SET template=? WHERE id=?", updates)
        conn.executescript("CREATE INDEX events_partition ON events(split,segment,ts,id);")
        graphs, label_counts = Counter(), Counter()
        for split in ("train", "validation", "test"):
            segments = conn.execute("SELECT segment,MIN(ts),MAX(ts) FROM events WHERE split=? GROUP BY segment", (split,)).fetchall()
            for segment, first, last in segments:
                start = float(np.floor(first/domain.stride_seconds)*domain.stride_seconds)
                while start <= last:
                    codes, history, edges = array("I"), deque(maxlen=cfg.data.order), set()
                    first_event, last_event, label = None, None, 0
                    for ts, slot, value in conn.execute(
                            "SELECT ts,template,label FROM events WHERE split=? AND segment=? AND ts>=? AND ts<? ORDER BY ts,id",
                            (split, segment, start, start+domain.window_seconds)):
                        first_event = ts if first_event is None else first_event
                        last_event, label = ts, max(label, value)
                        for previous in history:
                            edges.add((previous, slot))
                        history.append(slot)
                        codes.append(slot)
                    if codes:
                        if sys.byteorder != "little":
                            codes.byteswap()
                        conn.execute("INSERT INTO graphs(split,start,end,window_start,segment,label,edges,events,event_count) "
                                     "VALUES (?,?,?,?,?,?,?,?,?)", (split, first_event, last_event, start, segment, label,
                                      json.dumps(sorted(edges)), zlib.compress(codes.tobytes()), len(codes)))
                        graphs[split] += 1
                        label_counts[f"{split}:{label}"] += 1
                    nxt = conn.execute("SELECT MIN(ts) FROM events WHERE split=? AND segment=? AND ts>=?",
                                       (split, segment, start+domain.stride_seconds)).fetchone()[0]
                    if nxt is None:
                        break
                    earliest = np.floor((nxt-domain.window_seconds)/domain.stride_seconds+1)*domain.stride_seconds
                    start = float(max(start+domain.stride_seconds, earliest))
                conn.commit()
        conn.executescript("DROP TABLE events; CREATE INDEX graph_partition ON graphs(split,segment,window_start,id);")
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
        training_ids = sorted(training_counts, key=lambda item: (-training_counts[item], vocabulary[item]))
        metadata = {"identity": cache_identity(cfg, role), "raw_lines": count, "templates": len(vocabulary),
                    "training_templates": len(training_ids), "graphs": dict(graphs),
                    "label_counts_for_audit_only": dict(label_counts), "unmatched_raw_events_by_split": dict(unmatched),
                    "event_encoding": "zlib_little_endian_uint32", "candidate_selection_uses": "history_and_train_counts_only"}
        (folder/"vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
        (folder/"training_template_ids.json").write_text(json.dumps(training_ids), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        working.replace(folder/"graphs.sqlite")
        return metadata
    finally:
        conn.close()


def encode_cached(cfg, role):
    from sentence_transformers import SentenceTransformer

    folder = path_for(cfg, role)
    if not (folder/"graphs.sqlite").is_file():
        raise FileNotFoundError("Prepare this domain before encoding")
    vocabulary = json.loads((folder/"vocabulary.json").read_text(encoding="utf-8"))
    meta = {"encoder": cfg.semantics.model, "dimension": cfg.model.input_dim,
            "templates": len(vocabulary), "frozen_inference_only": True}
    feature_path, meta_path = folder/"features.npy", folder/"semantics.json"
    if feature_path.exists():
        if meta_path.exists() and json.loads(meta_path.read_text()) == meta:
            return meta
        raise FileExistsError("Incomplete or incompatible feature cache; inspect it before replacing")
    encoder = SentenceTransformer(cfg.semantics.model, local_files_only=cfg.semantics.local_files_only,
                                  device=cfg.semantics.device)
    temporary = folder/"features.partial.npy"
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                      shape=(len(vocabulary), cfg.model.input_dim))
    for start in range(0, len(vocabulary), cfg.semantics.batch_size):
        block = encoder.encode(vocabulary[start:start+cfg.semantics.batch_size], normalize_embeddings=True,
                               batch_size=cfg.semantics.batch_size, show_progress_bar=False)
        if block.shape[1] != cfg.model.input_dim or not np.isfinite(block).all():
            raise ValueError("Invalid semantic feature dimension or non-finite values")
        values[start:start+len(block)] = block
    values.flush()
    del values
    temporary.replace(feature_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
