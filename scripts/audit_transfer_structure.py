"""Read-only structural audit of a completed run; no training or test-label access."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from mtg_gan.config import load_config
from mtg_gan.data import domain_dir, read_graph_db
from mtg_gan.dataset import GraphDataset, collate_graphs
from mtg_gan.losses import laplacian, spectral_loss
from mtg_gan.training import device_for, load_checkpoint, move


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    vocab_sizes = {role: len(json.loads((domain_dir(cfg, role) / "vocabulary.json").read_text(encoding="utf-8")))
                   for role in ("source", "target")}
    initial = torch.load(Path(cfg.output_dir) / "pretrain.pt", map_location="cpu", weights_only=True)
    initial_decoder = initial["model"]["generator.decoder.2.weight"].clone()
    initial_bias = initial["model"]["generator.decoder.2.bias"].clone()
    initial_mask = initial["model"]["generator.mask.second.weight"].clone()
    del initial
    gc.collect()
    model = load_checkpoint(cfg, "source").eval()
    source_n, target_n = vocab_sizes["source"], vocab_sizes["target"]
    end = min(target_n, cfg.data.max_nodes)
    result = {"config": str(args.config.resolve()), "checkpoint_stage": "source",
              "diagnostic_split": "train", "uses_labels": False,
              "vocabulary_sizes": vocab_sizes,
              "target_slots_outside_source_support": max(0, target_n-source_n),
              "target_slots_outside_source_support_fraction": max(0, target_n-source_n)/target_n,
              "critic_parameters": sum(p.numel() for p in model.discriminator.parameters()),
              "generator_parameters": sum(p.numel() for p in model.generator.parameters())}
    if end > source_n:
        result["new_target_decoder_rows_unchanged_since_pretrain"] = bool(torch.equal(
            initial_decoder[source_n:end], model.generator.decoder[2].weight.cpu()[source_n:end]))
        result["new_target_decoder_bias_unchanged_since_pretrain"] = bool(torch.equal(
            initial_bias[source_n:end], model.generator.decoder[2].bias.cpu()[source_n:end]))
        result["new_target_mask_rows_unchanged_since_pretrain"] = bool(torch.equal(
            initial_mask[source_n:end], model.generator.mask.second.weight.cpu()[source_n:end]))
    torch.manual_seed(31415)
    result["domains"] = {}
    for role in ("source", "target"):
        dataset = GraphDataset(cfg, role, "train")
        n = vocab_sizes[role]
        rows = []
        for index in sorted({0, len(dataset)//2, len(dataset)-1}):
            batch = move(collate_graphs([dataset[index]]), device_for(cfg))
            output = model.generator(batch["features"], batch["adjacency"], batch["valid"], stochastic=False)
            prediction = output["prediction"]
            permutation = torch.cat((torch.randperm(n), torch.arange(n, cfg.data.max_nodes))).to(prediction.device)
            adjacency = batch["adjacency"].index_select(-2, permutation).index_select(-1, permutation)
            features = batch["features"].index_select(-2, permutation)
            permuted = model.generator(features, adjacency, batch["valid"], stochastic=False)["prediction"]
            expected = prediction.index_select(-2, permutation).index_select(-1, permutation)
            real = batch["target"]
            critic_original = model.discriminator(real)
            critic_permuted = model.discriminator(real.index_select(-2, permutation).index_select(-1, permutation))
            row = {"dataset_index": index, "graph_id": batch["graph_id"][0],
                   "generator_permutation_relative_error": float((permuted-expected).norm()/expected.norm().clamp_min(1e-12)),
                   "critic_permutation_absolute_error": float((critic_permuted-critic_original).abs()[0]),
                   "observed_edge_density": float(real[0, :n, :n].mean()),
                   "predicted_edge_probability_mean": float(prediction[0, :n, :n].mean())}
            if n > source_n:
                row["predicted_probability_old_destinations"] = float(prediction[0, :n, :source_n].mean())
                row["predicted_probability_new_destinations"] = float(prediction[0, :n, source_n:n].mean())
            rows.append(row)
            if role == "target" and index == 0:
                with torch.enable_grad():
                    probe = prediction.detach().clone().requires_grad_(True)
                    loss = spectral_loss(real, probe, batch["valid"], cfg.train.heat_time, cfg.train.spectral_trace)
                    gradient = torch.autograd.grad(loss, probe)[0]
                heat = torch.matrix_exp(-cfg.train.heat_time * laplacian(prediction[0, :n, :n]))
                result["target_spectral_probe"] = {
                    "graph_id": batch["graph_id"][0], "loss": float(loss.detach()),
                    "gradient_frobenius_norm": float(gradient.norm()),
                    "gradient_absolute_max": float(gradient.abs().max()),
                    "predicted_heat_trace": float(heat.trace()),
                    "predicted_heat_distance_from_uniform_projector": float((heat-torch.full_like(heat, 1/n)).norm())}
        dataset.close()
        with read_graph_db(domain_dir(cfg, role) / "graphs.sqlite") as db:
            intervals = db.execute("SELECT start,end FROM graphs WHERE split='train' ORDER BY start,id").fetchall()
        gaps = [max(0., intervals[i][0]-intervals[i-1][1]) for i in range(1, len(intervals))]
        result["domains"][role] = {"samples": rows,
                                   "largest_inter_graph_gap_hours": max(gaps, default=0.)/3600,
                                   "gaps_longer_than_window": sum(gap > getattr(cfg, role).window_seconds for gap in gaps)}
    path = Path(cfg.output_dir) / "structural_audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
