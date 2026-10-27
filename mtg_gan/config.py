"""Configuration: unspecified paper settings are explicit reconstruction choices."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path


@dataclass
class DomainConfig:
    name: str
    log_path: str
    label_path: str | None = None
    window_seconds: float = 3600
    stride_seconds: float = 3600
    max_lines: int | None = None
    allow_partial_sessions: bool = False
    sampling_manifest: str | None = None


@dataclass
class DataConfig:
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    context_graphs: int = 4
    order: int = 2
    cumulative: bool = True
    max_nodes: int = 1152
    drain_depth: int = 4
    drain_similarity: float = 0.4
    cache_root: str | None = None
    candidate_nodes: int = 512
    max_context_gap_seconds: float | None = None


@dataclass
class SemanticConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_files_only: bool = True
    batch_size: int = 64
    device: str = "cpu"


@dataclass
class ModelConfig:
    input_dim: int = 384
    hidden_dim: int = 64
    gcn_layers: int = 2
    tcn_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.1
    noise_dim: int = 16
    critic_hidden: int = 64
    mask_rate: float = 0.6
    mask_temperature: float = 1.0
    alpha_start: float = 0.1
    alpha_end: float = 0.9
    alpha_gamma: float = 2.0
    alpha_beta: float = 2.0
    architecture: str = "legacy"


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    batch_size: int = 2
    pretrain_epochs: int = 20
    source_epochs: int = 50
    target_epochs: int = 10
    learning_rate: float = 0.0001
    critic_steps: int = 3
    gradient_penalty: float = 10.0
    temperature: float = 0.2
    augment_drop: float = 0.2
    contrast_weight: float = 1.0
    contrast_margin: float = 1.0
    contrast_mode: str = "paper_separation"
    spectral_weight: float = 1.0
    spectral_trace: float = 0.01
    heat_time: float = 1.0
    reconstruction_weight: float = 0.0
    freeze_encoder: bool = False
    source_checkpoint_dir: str | None = None
    critic_learning_rate: float = 0.00005
    warmup_epochs: int = 0
    mask_budget_weight: float = 0.0
    spectral_gradient_probe_interval: int = 100
    source_validation: bool = False
    early_stopping_patience: int = 5


@dataclass
class ScoreConfig:
    local_weight: float = 0.5
    top_k: int = 3
    local_radius: int = 5
    quantile: float = 0.99
    dynamic_history: int = 256
    dynamic_min_history: int = 32
    dynamic: bool = True
    calibration: str = "source_normal_validation"
    component_normalization: str = "none"
    novelty_weight: float = 0.0


@dataclass
class Config:
    source: DomainConfig
    target: DomainConfig
    output_dir: str = "runs/bgl_to_thunderbird"
    data: DataConfig = field(default_factory=DataConfig)
    semantics: SemanticConfig = field(default_factory=SemanticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)

    def validate(self) -> None:
        d, m, t, s = self.data, self.model, self.train, self.score
        if not (0 < d.train_fraction < 1 and 0 < d.validation_fraction < 1 - d.train_fraction):
            raise ValueError("Train/validation fractions must leave a nonempty test fraction")
        positive = [d.context_graphs, d.order, d.max_nodes, m.input_dim, m.hidden_dim,
                    m.gcn_layers, m.tcn_layers, m.kernel_size, m.critic_hidden,
                    t.batch_size, t.critic_steps, s.top_k, s.dynamic_min_history,
                    self.semantics.batch_size, s.dynamic_history]
        if min(positive) < 1 or m.noise_dim < 0:
            raise ValueError("Sizes/counts must be positive (noise_dim may be zero)")
        if not (0 <= m.mask_rate <= 1 and 0 <= m.dropout < 1 and 0 <= t.augment_drop < 1):
            raise ValueError("Invalid mask/drop probability")
        if not (0 <= m.alpha_start <= m.alpha_end <= 1) or m.alpha_gamma <= 0 or m.alpha_beta < 0:
            raise ValueError("Invalid mask schedule")
        if min(m.mask_temperature, t.temperature, t.learning_rate, t.heat_time) <= 0:
            raise ValueError("Temperatures, heat time and learning rate must be positive")
        if min(t.pretrain_epochs, t.source_epochs, t.target_epochs) < 0:
            raise ValueError("Epoch counts cannot be negative")
        if min(t.gradient_penalty, t.spectral_weight, t.spectral_trace,
               t.contrast_weight, t.reconstruction_weight, t.contrast_margin) < 0:
            raise ValueError("Loss weights and margin cannot be negative")
        if t.contrast_mode not in {"paper_separation", "invariance"}:
            raise ValueError("Unknown contrast_mode")
        if not (0 <= s.local_weight <= 1 and 0 < s.quantile < 1):
            raise ValueError("Invalid score weight/quantile")
        if s.local_radius < 0 or s.dynamic_min_history > s.dynamic_history:
            raise ValueError("Invalid dynamic threshold/local window settings")
        if s.calibration not in {"source_normal_validation", "target_unlabeled_train"}:
            raise ValueError("Unknown calibration policy")
        if m.architecture not in {"legacy", "inductive_v2"}:
            raise ValueError("Unknown model architecture")
        if d.candidate_nodes < 2 or t.warmup_epochs < 0 or t.critic_learning_rate <= 0 or t.mask_budget_weight < 0:
            raise ValueError("Invalid inductive training settings")
        if t.spectral_gradient_probe_interval < 1:
            raise ValueError("Spectral gradient probe interval must be positive")
        if t.early_stopping_patience < 1:
            raise ValueError("Early stopping patience must be positive")
        if m.architecture == "inductive_v2" and t.warmup_epochs and t.reconstruction_weight <= 0:
            raise ValueError("Prediction warmup requires positive reconstruction_weight")
        if d.max_context_gap_seconds is not None and d.max_context_gap_seconds <= 0:
            raise ValueError("Context gap limit must be positive")
        if s.component_normalization not in {"none", "robust"} or s.novelty_weight < 0:
            raise ValueError("Invalid score normalization/novelty weight")
        if m.architecture == "inductive_v2" and not d.cache_root:
            raise ValueError("inductive_v2 requires a shared data.cache_root")
        for domain in [self.source, self.target]:
            if m.architecture == "inductive_v2" and domain.name.lower() not in {"bgl", "spirit", "thunderbird"}:
                raise ValueError("inductive_v2 currently supports the three-domain benchmark")
            if domain.name.lower() not in {"bgl", "thunderbird", "hdfs", "spirit", "openstack"}:
                raise ValueError(f"Unsupported domain {domain.name}")
            if domain.window_seconds <= 0 or domain.stride_seconds <= 0:
                raise ValueError("Time window and stride must be positive")
            if domain.stride_seconds > domain.window_seconds:
                raise ValueError("Stride larger than window would silently drop log events")
            if domain.max_lines is not None and domain.max_lines <= 0:
                raise ValueError("max_lines must be positive or null")
            if domain.name.lower() == "hdfs" and not domain.label_path:
                raise ValueError("HDFS requires a block label CSV")
            if domain.name.lower() == "hdfs" and domain.max_lines and not domain.allow_partial_sessions:
                raise ValueError("HDFS line limits truncate sessions; explicitly allow_partial_sessions for debugging")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = {"source": DomainConfig, "target": DomainConfig, "data": DataConfig,
               "semantics": SemanticConfig, "model": ModelConfig, "train": TrainConfig,
               "score": ScoreConfig}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for name, cls in classes.items():
        if name in raw:
            raw[name] = cls(**raw[name])
    cfg = Config(**raw)
    # Relative paths are relative to the config file, independent of current cwd.
    for domain in [cfg.source, cfg.target]:
        for key in ["log_path", "label_path", "sampling_manifest"]:
            value = getattr(domain, key)
            if value is not None:
                setattr(domain, key, str((path.parent / value).resolve()))
    cfg.output_dir = str((path.parent / cfg.output_dir).resolve())
    if cfg.data.cache_root is not None:
        cfg.data.cache_root = str((path.parent / cfg.data.cache_root).resolve())
    if cfg.train.source_checkpoint_dir is not None:
        cfg.train.source_checkpoint_dir = str((path.parent / cfg.train.source_checkpoint_dir).resolve())
    if cfg.semantics.model.startswith(("./", "../", ".\\", "..\\")):
        cfg.semantics.model = str((path.parent / cfg.semantics.model).resolve())
    cfg.validate()
    return cfg
