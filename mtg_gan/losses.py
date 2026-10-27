"""Paper equations with their ambiguities made explicit (see docs/methodology.md)."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric node-level InfoNCE; include the positive in the denominator."""
    if len(first) < 2:
        return first.sum() * 0
    first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(len(first), device=first.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def perturbation_loss(original, masked, margin: float, mode: str = "paper_separation"):
    distance = torch.sqrt((original - masked).square().sum(-1) + 1e-8)
    if mode == "paper_separation":
        # Eq. (9) literally ENCOURAGES distance >= margin, despite prose saying close.
        return F.relu(margin - distance).mean() + 1e-8
    if mode == "invariance":
        return F.relu(distance - margin).mean() + 1e-8
    raise ValueError(f"Unknown perturbation mode {mode}")


def laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetrization is required to use a real orthonormal eigenspectrum."""
    adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2
    return torch.diag_embed(adjacency.sum(-1)) - adjacency


def spectral_loss(real, predicted, valid, heat_time: float, trace_weight: float):
    """Eq. (10); explicit Phi(L)=exp(-t L), a differentiable spectral filter.

    No eigenvector backpropagation at repeated eigenvalues. Padded nodes are
    excluded from both Frobenius and trace terms. Trace sign follows the paper.
    """
    terms = []
    for r, p, mask in zip(real, predicted, valid):
        r, p = r[mask][:, mask], p[mask][:, mask]
        phi_r = torch.matrix_exp(-heat_time * laplacian(r))
        phi_p = torch.matrix_exp(-heat_time * laplacian(p))
        terms.append((phi_r-phi_p).square().sum() + trace_weight * (phi_r * phi_p).sum())
    return torch.stack(terms).mean()


def gradient_penalty(critic, real, fake, valid):
    alpha = torch.rand((real.shape[0], 1, 1), device=real.device)
    interpolated = (alpha * real + (1-alpha) * fake.detach()).requires_grad_(True)
    result = critic(interpolated)
    gradient = torch.autograd.grad(result.sum(), interpolated, create_graph=True)[0]
    mask = valid[:, :, None] & valid[:, None, :]
    norm = (gradient * mask).flatten(1).norm(2, dim=1)
    return (norm - 1).square().mean()


def normalized_laplacian(adjacency):
    symmetric = (adjacency+adjacency.transpose(-1, -2))/2
    degree = symmetric.sum(-1)
    inverse = degree.clamp_min(1e-8).rsqrt()
    return inverse[..., :, None] * laplacian(adjacency) * inverse[..., None, :]


def normalized_spectral_loss(real, predicted, valid, heat_time, trace_weight):
    """Extension: bounded-frequency heat kernels, per-node Frobenius loss."""
    losses = []
    for r, p, mask in zip(real, predicted, valid):
        r, p = r[mask][:, mask], p[mask][:, mask]
        first = torch.matrix_exp(-heat_time*normalized_laplacian(r))
        second = torch.matrix_exp(-heat_time*normalized_laplacian(p))
        losses.append(((first-second).square().sum()+trace_weight*(first*second).sum())/max(1, len(r)))
    return torch.stack(losses).mean()


def balanced_edge_loss(logits, target, valid):
    """Equal positive/negative class means, then equal graph means."""
    losses = []
    for value, real, mask in zip(logits, target, valid):
        pairs = mask[:, None] & mask[None, :]
        raw = F.binary_cross_entropy_with_logits(value, real, reduction="none")
        groups = []
        for selected in (pairs & (real > .5), pairs & (real <= .5)):
            if selected.any():
                groups.append(raw[selected].mean())
        losses.append(torch.stack(groups).mean())
    return torch.stack(losses).mean()
