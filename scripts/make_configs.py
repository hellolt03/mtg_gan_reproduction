"""Generate example configs only; never read full datasets or train models."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtg_gan.config import Config, DataConfig, DomainConfig, ModelConfig


def main():
    base = Path("D:/PaperWork/count_granger/dataset")
    domains = {
        "bgl": DomainConfig("bgl", str(base / "BGL/BGL.log"), window_seconds=21600, stride_seconds=3600),
        "thunderbird": DomainConfig("thunderbird", str(base / "Thunderbird/Thunderbird.log")),
        "hdfs": DomainConfig("hdfs", str(base / "HDFS/HDFS.log"), str(base / "HDFS/anomaly_label.csv")),
        "openstack": DomainConfig("openstack", str(base / "OpenStack/OpenStack_labeled.log"), window_seconds=60, stride_seconds=60),
        "spirit": DomainConfig("spirit", str(base / "Spirit/Spirit1G.log")),
    }
    for source, target in [("bgl", "thunderbird"), ("thunderbird", "bgl"), ("bgl", "hdfs"),
                           ("hdfs", "bgl"), ("bgl", "openstack"), ("openstack", "bgl"), ("bgl", "spirit")]:
        name = f"{source}_to_{target}"
        # Capacities based on reported counts; this Drain configuration may find more.
        nodes = 1152 if "thunderbird" in (source, target) else 384
        if "spirit" in (source, target):
            nodes = 2048  # exploratory, actual vocabulary is unknown until preparation
        cfg = Config(domains[source], domains[target], output_dir=f"../runs/{name}",
                     data=DataConfig(max_nodes=nodes), model=ModelConfig())
        cfg.validate()
        (ROOT / "configs" / f"{name}.json").write_text(
            json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
