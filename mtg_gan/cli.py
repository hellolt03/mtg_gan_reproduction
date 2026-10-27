"""Explicit stages: no command implicitly launches a complete experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def inspect_inputs(cfg):
    from .data import parse_line

    dense_nodes = cfg.data.candidate_nodes if cfg.model.architecture == "inductive_v2" else cfg.data.max_nodes
    result = {"output_dir": cfg.output_dir, "architecture": cfg.model.architecture,
              "dense_nodes_per_example_limit": dense_nodes, "domains": {}, "dense_batch_estimate_mib":
              round(cfg.train.batch_size * (cfg.data.context_graphs + 1) * dense_nodes**2 * 4 / 2**20, 1),
              "note": "Adjacency inputs only; excludes gradients, masks and spectral workspaces"}
    for role in ("source", "target"):
        domain = getattr(cfg, role)
        path = Path(domain.log_path)
        row = {"system": domain.name, "path": str(path), "exists": path.is_file()}
        if path.is_file():
            row["bytes"] = path.stat().st_size
            with path.open(encoding="utf-8", errors="replace") as stream:
                first = parse_line(domain.name, next(stream))
            row["first_timestamp"] = first.timestamp
            row["first_message"] = first.message
        if domain.label_path:
            row["labels_exist"] = Path(domain.label_path).is_file()
        result["domains"][role] = row
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="MTG-GAN reconstruction; see docs/methodology.md for assumptions")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare", "encode", "pretrain", "train-source", "adapt-target", "calibrate", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in {"prepare", "encode"}:
            command.add_argument("--role", choices=["source", "target", "both"], default="both")
        if name == "train-source":
            command.add_argument("--skip-pretrain", action="store_true", help="Explicit GCL ablation")
        if name in {"calibrate", "evaluate"}:
            command.add_argument("--stage", choices=["source", "target"], default="target")
        if name == "evaluate":
            command.add_argument("--role", choices=["source", "target"], default="target")
            command.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "inspect":
        result = inspect_inputs(cfg)
    elif args.command in {"prepare", "encode"}:
        from .data import encode_domain, prepare_domain
        action = prepare_domain if args.command == "prepare" else encode_domain
        roles = ("source", "target") if args.role == "both" else (args.role,)
        result = {role: action(cfg, role) for role in roles}
    else:
        from .training import calibrate, evaluate, pretrain, train_gan
        if args.command == "pretrain":
            result = pretrain(cfg)
        elif args.command == "train-source":
            result = train_gan(cfg, skip_pretrain=args.skip_pretrain)
        elif args.command == "adapt-target":
            result = train_gan(cfg, target=True)
        elif args.command == "calibrate":
            result = calibrate(cfg, args.stage)
        else:
            result = evaluate(cfg, args.role, args.stage, args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))
