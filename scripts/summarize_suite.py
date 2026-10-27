"""Aggregate completed runs only; missing directions are explicitly retained."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import permutations
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"benchmarks/bgl_spirit_thunderbird/v2_completed_runs.json")
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError("No completed v2 runs; no performance results can be reported")
    latest = {}
    for row in json.loads(args.input.read_text(encoding="utf-8")):
        latest[(row["source"], row["target"], row["seed"])] = row
    grouped = defaultdict(list)
    for row in latest.values():
        grouped[(row["source"], row["target"])].append(row)
    directions = []
    for source, target in permutations(("bgl", "spirit", "thunderbird"), 2):
        rows = grouped[(source, target)]
        summary = {"source": source, "target": target, "completed_seeds": sorted(r["seed"] for r in rows)}
        for phase in ("before", "after"):
            scores = {}
            for metric in ("precision", "recall", "f1", "roc_auc", "average_precision", "predicted_positive_rate"):
                values = [r[phase][metric] for r in rows if r[phase].get(metric) is not None]
                scores[metric] = {"count": len(values), "mean": statistics.mean(values) if values else None,
                                   "std": statistics.stdev(values) if len(values) > 1 else None}
            summary[phase] = scores
        directions.append(summary)
    available = [d["after"]["f1"]["mean"] for d in directions if d["after"]["f1"]["mean"] is not None]
    balanced = len({tuple(d["completed_seeds"]) for d in directions}) == 1
    report = {"directions": directions, "completed_directions": len(available), "balanced_seed_coverage": balanced,
              "six_direction_macro_f1": statistics.mean(available) if len(available) == 6 and balanced else None,
              "note": "No six-direction macro score is emitted if any direction is missing"}
    output = args.input.with_name("v2_aggregate.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
