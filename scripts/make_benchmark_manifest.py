"""Declare the six requested transfer tasks; does not start training."""
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/PaperWork/count_granger/dataset"))
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/bgl_spirit_thunderbird/manifest.json")
    args = parser.parse_args()
    filenames = {"bgl": "BGL/BGL.log", "spirit": "Spirit/Spirit1G.log",
                 "thunderbird": "Thunderbird/Thunderbird.log"}
    domains = {}
    for name, filename in filenames.items():
        path = (args.data_root / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        domains[name] = {"log_path": str(path), "bytes": path.stat().st_size,
                         "planned_scope": "entire_available_file",
                         "window_seconds": 21600 if name == "bgl" else 3600,
                         "stride_seconds": 3600,
                         "window_setting_status": "extension_choice" if name == "spirit" else "current_paper_reconstruction"}
    tasks = [{"id": f"{source}_to_{target}", "source": source, "target": target,
              "status": "awaiting_user_approval_before_validation_and_training", "source_checkpoint_key": source,
              "config": str(ROOT / "configs" / "inductive_v2" / f"{source}_to_{target}.json"),
              "evaluations": ["source_test", "target_before_adaptation", "target_after_adaptation"]}
             for source, target in permutations(domains, 2)]
    manifest = {
        "suite": "bgl_spirit_thunderbird_six_directions", "status": "code_written_waiting_user_approval",
        "domains": domains, "tasks": tasks, "planned_seeds": [42, 43, 44],
        "split_policy": {"unit": "chronological_raw_records_before_windowing", "fractions": [0.6, 0.2, 0.2],
                         "reuse_same_domain_split_across_all_tasks": True,
                         "context_must_not_cross_sampling_segments": True},
        "source_training": "known_normal_samples_only",
        "target_training": "unlabeled_train_split_only",
        "primary_calibration": "source_normal_validation",
        "secondary_calibration": "target_unlabeled_train_report_separately",
        "source_checkpoint_reuse": "three source checkpoints per seed, two independent target copies each",
        "implemented_model_track": "inductive_v2_extension_not_runtime_validated",
        "runner": str(ROOT / "scripts" / "run_six_directions.py"),
        "runtime_validation": "not_run_user_requested_approval_first",
        "required_runtime_validation": ["continuous_data_and_shared_domain_cache", "shared_pair_decoder_and_mask",
                          "permutation_invariant_graph_critic", "node_presence_and_sparse_prediction",
                          "size_aware_spectral_scoring_and_component_calibration", "source_and_adaptation_validation"],
        "previous_diagnostic_run": str(ROOT / "runs/bgl_to_thunderbird_pilot_20261026"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Declared {len(tasks)} transfer directions: {args.output.resolve()}")
    for task in tasks:
        print(task["id"], task["status"])


if __name__ == "__main__":
    main()
