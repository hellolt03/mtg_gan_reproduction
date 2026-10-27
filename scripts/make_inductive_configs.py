"""Write six v2 configs only. Does not prepare data or run any model."""
from itertools import permutations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mtg_gan.config import Config, DomainConfig, DataConfig, SemanticConfig, ModelConfig, TrainConfig, ScoreConfig


def main():
    folder = ROOT/"configs/inductive_v2"
    folder.mkdir(parents=True, exist_ok=True)
    base = Path("D:/PaperWork/count_granger/dataset")
    files = {"bgl": "BGL/BGL.log", "spirit": "Spirit/Spirit1G.log", "thunderbird": "Thunderbird/Thunderbird.log"}
    domains = {name: DomainConfig(name, str(base/path), window_seconds=21600 if name == "bgl" else 3600,
                                  stride_seconds=3600) for name, path in files.items()}
    for source, target in permutations(domains, 2):
        name = f"{source}_to_{target}"
        cfg = Config(domains[source], domains[target], output_dir=f"../../runs/inductive_v2/{name}/seed42",
                     data=DataConfig(cache_root="../../prepared/inductive_v2", max_nodes=50000,
                                     candidate_nodes=512, max_context_gap_seconds=5400),
                     semantics=SemanticConfig(model="../../models/all-MiniLM-L6-v2", batch_size=32, device="cpu"),
                     model=ModelConfig(architecture="inductive_v2", mask_rate=.4, noise_dim=8),
                     train=TrainConfig(batch_size=1, pretrain_epochs=10, source_epochs=20, target_epochs=5,
                                       source_checkpoint_dir=f"../../source_runs/inductive_v2/{source}/seed42",
                                       warmup_epochs=3, critic_learning_rate=.00005,
                                       contrast_mode="invariance", contrast_margin=.1, contrast_weight=.1,
                                       spectral_weight=.1, spectral_trace=0., reconstruction_weight=1.,
                                       mask_budget_weight=.1, source_validation=True),
                     score=ScoreConfig(dynamic=False, component_normalization="robust", novelty_weight=.25))
        cfg.validate()
        (folder/f"{name}.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
