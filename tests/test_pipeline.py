import json

from mtg_gan.data import domain_dir
from mtg_gan.training import calibrate, evaluate, pretrain, train_gan


def test_synthetic_full_pipeline_without_real_experiment(prepared):
    cfg = prepared
    assert pretrain(cfg)[0]["node_pairs"] > 0
    assert train_gan(cfg)[0]["spectral"] >= 0
    train_gan(cfg, target=True)
    calibration = calibrate(cfg)
    assert calibration["policy"] == "source_normal_validation"
    report = evaluate(cfg)
    assert report["count"] > 0 and 0 <= report["f1"] <= 1
    from pathlib import Path
    assert (Path(cfg.output_dir) / "target_target_test_predictions.csv").is_file()
    cfg.score.calibration = "target_unlabeled_train"
    assert calibrate(cfg)["policy"] == "target_unlabeled_train"
