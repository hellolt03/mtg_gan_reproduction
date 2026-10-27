"""Six-direction v2 runner. Default is a plan; --execute starts real work.

The current user requested approval before experiments. Do not invoke --execute
until that approval arrives. This script does not itself solicit approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mtg_gan.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT/"configs/inductive_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--execute", action="store_true", help="Run preparation, training and evaluation")
    args = parser.parse_args()
    config_paths = sorted(args.config_dir.glob("*.json"))
    pairs = [load_config(path) for path in config_paths]
    expected = {(a, b) for a in ("bgl", "spirit", "thunderbird") for b in ("bgl", "spirit", "thunderbird") if a != b}
    if len(pairs) != 6 or {(c.source.name, c.target.name) for c in pairs} != expected:
        raise ValueError("Config directory must contain exactly the six requested directed pairs")
    if any(c.model.architecture != "inductive_v2" for c in pairs):
        raise ValueError("This runner is only for the v2 extension")
    print(json.dumps({"execute": args.execute, "seeds": args.seeds,
                      "directions": [f"{c.source.name}_to_{c.target.name}" for c in pairs],
                      "source_models_per_seed": 3, "target_adaptations_per_seed": 6,
                      "execution": "serial", "note": "No work starts without --execute"}, indent=2), flush=True)
    if not args.execute:
        return
    from mtg_gan.data import prepare_domain, encode_domain, domain_dir
    from mtg_gan.training import (pretrain, train_gan, calibrate, evaluate,
                                  checkpoint_path, load_checkpoint)

    prepared = set()
    for cfg in pairs:
        for role in ("source", "target"):
            key = str(domain_dir(cfg, role))
            if key not in prepared:
                prepare_domain(cfg, role)
                encode_domain(cfg, role)
                prepared.add(key)
    output = ROOT/"benchmarks/bgl_spirit_thunderbird/v2_completed_runs.json"
    rows = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    for seed in args.seeds:
        configs = []
        for path in config_paths:
            cfg = load_config(path)
            cfg.train.seed = seed
            cfg.train.source_checkpoint_dir = str(Path(cfg.train.source_checkpoint_dir).parent/f"seed{seed}")
            cfg.output_dir = str(Path(cfg.output_dir).parent/f"seed{seed}")
            configs.append(cfg)
        trained_sources = set()
        for cfg in configs:
            if cfg.source.name not in trained_sources:
                if checkpoint_path(cfg, "source").exists():
                    restored = load_checkpoint(cfg, "source")
                    del restored
                else:
                    if not checkpoint_path(cfg, "pretrain").exists():
                        pretrain(cfg)
                    train_gan(cfg)
                trained_sources.add(cfg.source.name)
            calibrate(cfg, "source")
            source_result = evaluate(cfg, "source", "source")
            before = evaluate(cfg, "target", "source")
            if checkpoint_path(cfg, "target").exists():
                restored = load_checkpoint(cfg, "target")
                del restored
            else:
                train_gan(cfg, target=True)
            calibrate(cfg, "target")
            after = evaluate(cfg, "target", "target")
            rows.append({"source": cfg.source.name, "target": cfg.target.name, "seed": seed,
                         "run_dir": cfg.output_dir, "source_metrics": source_result,
                         "before": before, "after": after})
            output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Finished requested seed/direction instances; run summarize_suite.py for aggregate metrics.", flush=True)


if __name__ == "__main__":
    main()
