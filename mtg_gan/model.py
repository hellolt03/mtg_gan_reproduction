"""GCN -> causal TCN -> adaptive masking -> next-adjacency prediction."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .config import Config, ModelConfig


def pair_mask(valid: torch.Tensor) -> torch.Tensor:
    return valid.unsqueeze(-1) & valid.unsqueeze(-2)


def normalized_adjacency(adjacency: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # Symmetric GCN propagation, while stored and predicted edges remain directed.
    adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2
    n = adjacency.shape[-1]
    adjacency = adjacency + torch.eye(n, device=adjacency.device, dtype=adjacency.dtype)
    adjacency = adjacency * pair_mask(valid).to(adjacency.dtype)
    inv = adjacency.sum(-1).clamp_min(1e-8).rsqrt()
    return inv.unsqueeze(-1) * adjacency * inv.unsqueeze(-2)


class GraphEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dims = [cfg.input_dim] + [cfg.hidden_dim] * cfg.gcn_layers
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = cfg.dropout

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        norm = normalized_adjacency(adjacency, valid)
        value = features
        for layer in self.layers:
            value = F.relu(layer(norm @ value))
            value = F.dropout(value, self.dropout, self.training)
            value = value * valid.unsqueeze(-1)
        return value


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu(value + self.dropout(self.conv(F.pad(value, (self.padding, 0)))))


class TemporalEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.Sequential(*[CausalBlock(cfg.hidden_dim, cfg.kernel_size, 2**i, cfg.dropout)
                                      for i in range(cfg.tcn_layers)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Per-node temporal trajectories: B,W,N,H -> (B*N),H,W.
        b, w, n, h = value.shape
        value = value.permute(0, 2, 3, 1).reshape(b * n, h, w)
        return self.blocks(value).reshape(b, n, h, w).permute(0, 3, 1, 2)


class AdaptiveMask(nn.Module):
    def __init__(self, cfg: ModelConfig, nodes: int):
        super().__init__()
        self.cfg = cfg
        self.first = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2)
        self.norm = nn.LayerNorm(cfg.hidden_dim * 2)
        self.second = nn.Linear(cfg.hidden_dim * 2, nodes)

    def weight(self, progress: float) -> float:
        p = min(1., max(0., progress))
        c = self.cfg
        # Eq. (8), including its non-monotonic exponential term.
        return c.alpha_start + p**c.alpha_gamma * (c.alpha_end-c.alpha_start) * math.exp(-c.alpha_beta * abs(p-.5))

    def forward(self, value: torch.Tensor, valid: torch.Tensor, progress: float,
                stochastic: bool | None = None) -> torch.Tensor:
        stochastic = self.training if stochastic is None else stochastic
        c = self.cfg
        logits = self.second(self.norm(F.relu(self.first(value)))) / math.sqrt(c.hidden_dim)
        if c.mask_rate == 0:
            return torch.ones_like(logits) * pair_mask(valid)[:, None]
        if c.mask_rate == 1:
            return torch.zeros_like(logits)
        logits = logits + math.log(c.mask_rate / (1-c.mask_rate))
        if stochastic:
            # Binary Gumbel-Softmax per directed edge; 1 means DELETE.
            learned = F.gumbel_softmax(torch.stack((torch.zeros_like(logits), logits), -1),
                                       tau=c.mask_temperature, hard=True, dim=-1)[..., 1]
            random = (torch.rand_like(logits) < c.mask_rate).to(logits.dtype)
        else:
            # Expected mask makes repeated scoring deterministic.
            learned = torch.sigmoid(logits)
            random = torch.full_like(logits, c.mask_rate)
        alpha = self.weight(progress)
        return (1 - alpha * learned - (1-alpha) * random) * pair_mask(valid)[:, None]


class Generator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = GraphEncoder(cfg.model)
        self.temporal = TemporalEncoder(cfg.model)
        self.mask = AdaptiveMask(cfg.model, cfg.data.max_nodes)
        self.decoder = nn.Sequential(nn.Linear(cfg.model.hidden_dim + cfg.model.noise_dim, cfg.model.hidden_dim),
                                     nn.ReLU(), nn.Linear(cfg.model.hidden_dim, cfg.data.max_nodes))

    def encode_history(self, features, adjacency, valid):
        w = adjacency.shape[1]
        spatial = self.encoder(features[:, None].expand(-1, w, -1, -1), adjacency, valid[:, None])
        return self.temporal(spatial)

    def forward(self, features, adjacency, valid, *, progress: float = 1., noise=None,
                stochastic: bool | None = None) -> dict:
        original = self.encode_history(features, adjacency, valid)
        keep = self.mask(original, valid, progress, stochastic)
        masked = self.encode_history(features, adjacency * keep, valid)
        latest = masked[:, -1]
        stochastic = self.training if stochastic is None else stochastic
        if noise is None:
            shape = (*latest.shape[:-1], self.cfg.model.noise_dim)
            noise = torch.randn(shape, device=latest.device) if stochastic else latest.new_zeros(shape)
        prediction = torch.sigmoid(self.decoder(torch.cat((latest, noise), -1)))
        prediction = prediction * pair_mask(valid)
        denominator = valid.sum(-1, keepdim=True).clamp_min(1)
        return {"prediction": prediction, "keep_mask": keep,
                "original_features": (original[:, -1] * valid[..., None]).sum(-2) / denominator,
                "masked_features": (masked[:, -1] * valid[..., None]).sum(-2) / denominator}


class Discriminator(nn.Module):
    """Eq. (11), unbounded Wasserstein critic. Lower values mean more normal.

    This convention follows the MINIMIZATION in Eqs. (12)-(13).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        n, h = cfg.data.max_nodes, cfg.model.critic_hidden
        self.network = nn.Sequential(nn.Linear(n*n, h), nn.LeakyReLU(.2),
                                     nn.Linear(h, h), nn.LeakyReLU(.2), nn.Linear(h, 1))

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        return self.network(adjacency.flatten(-2)).squeeze(-1)


class MTGGAN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)
