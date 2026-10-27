"""Inductive extension: no learnable parameter is indexed by a template ID.

This changes the paper reconstruction's FFN decoder/critic and is reported as
an extension, not as a verified restoration of the authors' implementation.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import GraphEncoder, TemporalEncoder, pair_mask


def masked_pool(value, weights):
    weights = weights.to(value.dtype)
    return (value * weights[..., None]).sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1)


class SharedPairMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.project = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.LayerNorm(h))
        self.query = nn.Linear(h, h, bias=False)
        self.key = nn.Linear(h, h, bias=False)

    def weight(self, progress):
        c = self.cfg
        p = min(1., max(0., progress))
        return c.alpha_start + p**c.alpha_gamma * (c.alpha_end-c.alpha_start) * math.exp(-c.alpha_beta * abs(p-.5))

    def forward(self, value, valid, progress, stochastic):
        c = self.cfg
        value = self.project(value)
        logits = self.query(value) @ self.key(value).transpose(-1, -2) / math.sqrt(c.hidden_dim)
        pairs = pair_mask(valid)[:, None]
        if c.mask_rate == 0:
            return torch.ones_like(logits) * pairs
        if c.mask_rate == 1:
            return torch.zeros_like(logits)
        logits = logits + math.log(c.mask_rate/(1-c.mask_rate))
        if stochastic:
            learned = F.gumbel_softmax(torch.stack((torch.zeros_like(logits), logits), -1),
                                       tau=c.mask_temperature, hard=True, dim=-1)[..., 1]
            random = (torch.rand_like(logits) < c.mask_rate).to(logits.dtype)
        else:
            learned = logits.sigmoid()
            random = torch.full_like(logits, c.mask_rate)
        alpha = self.weight(progress)
        return (1-alpha*learned-(1-alpha)*random) * pairs


class InductiveGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg.model
        self.encoder = GraphEncoder(c)
        self.temporal = TemporalEncoder(c)
        self.semantic = nn.Linear(c.input_dim, c.hidden_dim)
        self.mask = SharedPairMask(c)
        self.decode = nn.Sequential(nn.Linear(c.hidden_dim+c.noise_dim, c.hidden_dim), nn.ReLU(),
                                    nn.LayerNorm(c.hidden_dim))
        self.query = nn.Linear(c.hidden_dim, c.hidden_dim, bias=False)
        self.key = nn.Linear(c.hidden_dim, c.hidden_dim, bias=False)
        self.edge_bias = nn.Parameter(torch.tensor(-3.0))

    def encode_history(self, features, adjacency, present):
        x = features[:, None].expand(-1, adjacency.shape[1], -1, -1)
        return self.temporal(self.encoder(x, adjacency, present))

    def forward(self, features, adjacency, valid, *, present=None, progress=1., noise=None, stochastic=None):
        stochastic = self.training if stochastic is None else stochastic
        if present is None:
            raise ValueError("inductive_v2 requires event-derived history node presence")
        original = self.encode_history(features, adjacency, present)
        semantic = torch.tanh(self.semantic(features))
        keep = self.mask(original+semantic[:, None], valid, progress, stochastic)
        masked = self.encode_history(features, adjacency*keep, present)
        latest = masked[:, -1] + semantic
        if noise is None:
            shape = (*latest.shape[:-1], self.cfg.model.noise_dim)
            noise = torch.randn(shape, device=latest.device) if stochastic else latest.new_zeros(shape)
        value = self.decode(torch.cat((latest, noise), -1))
        logits = self.query(value) @ self.key(value).transpose(-1, -2) / math.sqrt(value.shape[-1])
        logits = logits + self.edge_bias
        prediction = logits.sigmoid() * pair_mask(valid)
        history_nodes = present.any(dim=1)
        existing = adjacency > 0
        deleted_rate = ((1-keep)*existing).sum() / existing.sum().clamp_min(1)
        return {"prediction": prediction, "logits": logits, "keep_mask": keep,
                "original_features": masked_pool(original[:, -1], history_nodes),
                "masked_features": masked_pool(masked[:, -1], history_nodes),
                "mask_budget": (deleted_rate-self.cfg.model.mask_rate).square()}


class InductiveCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.model.hidden_dim
        self.semantic = nn.Linear(cfg.model.input_dim, h)
        self.structure = nn.Linear(2, h)
        self.encoder = GraphEncoder(replace(cfg.model, input_dim=h, dropout=0.0))
        self.head = nn.Sequential(nn.Linear(2*h+2, cfg.model.critic_hidden), nn.LeakyReLU(.2),
                                  nn.Linear(cfg.model.critic_hidden, 1))

    def forward(self, adjacency, features, valid, *, present=None):
        pairs = pair_mask(valid)
        adjacency = adjacency * pairs
        out_degree, in_degree = adjacency.sum(-1), adjacency.sum(-2)
        degrees = torch.stack((out_degree.log1p(), in_degree.log1p()), -1)
        value = F.relu(self.semantic(features) + self.structure(degrees))
        value = self.encoder(value, adjacency, valid)
        # Continuous presence for fake/interpolated graphs preserves GP gradients.
        weights = -torch.expm1(-(out_degree+in_degree)) if present is None else present.to(value.dtype)
        weights = weights * valid
        mean = masked_pool(value, weights)
        maximum = (value*weights[..., None]).amax(-2)
        nodes = valid.sum(-1).clamp_min(1)
        density = adjacency.sum((-1, -2))/nodes.square()
        coverage = weights.sum(-1)/nodes
        return self.head(torch.cat((mean, maximum, density[:, None], coverage[:, None]), -1)).squeeze(-1)


class InductiveMTGGAN(nn.Module):
    inductive = True

    def __init__(self, cfg):
        super().__init__()
        self.generator = InductiveGenerator(cfg)
        self.discriminator = InductiveCritic(cfg)
