import torch

from mtg_gan.losses import info_nce, laplacian, perturbation_loss, spectral_loss
from mtg_gan.model import MTGGAN, TemporalEncoder
from mtg_gan.scoring import CausalThreshold, anomaly_scores


def batch(cfg):
    n = cfg.data.max_nodes
    a = torch.zeros(2, cfg.data.context_graphs, n, n)
    a[:, :, 0, 1] = 1
    a[:, :, 1, 2] = 1
    valid = torch.arange(n)[None].expand(2, -1) < 4
    return {"features": torch.randn(2, n, cfg.model.input_dim), "adjacency": a,
            "target": a[:, -1].clone(), "valid": valid,
            "event_ids": [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])]}


def test_tcn_is_causal(tiny_config):
    encoder = TemporalEncoder(tiny_config.model).eval()
    x = torch.randn(2, 7, 3, tiny_config.model.hidden_dim)
    first = encoder(x)
    x[:, 4:] += 200
    second = encoder(x)
    torch.testing.assert_close(first[:, :4], second[:, :4])


def test_mask_gradients_padding_and_determinism(tiny_config):
    model = MTGGAN(tiny_config).train()
    b = batch(tiny_config)
    out = model.generator(b["features"], b["adjacency"], b["valid"], progress=.5)
    assert out["prediction"].shape == b["target"].shape
    assert (out["prediction"][:, 4:] == 0).all()
    out["prediction"].sum().backward()
    grad = model.generator.mask.second.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    model.eval()
    first = anomaly_scores(model, b, tiny_config)
    second = anomaly_scores(model, b, tiny_config)
    torch.testing.assert_close(first["score"], second["score"])
    assert torch.isfinite(first["score"]).all()
    with torch.no_grad():
        prediction = model.generator(b["features"], b["adjacency"], b["valid"], stochastic=False)["prediction"]
        difference = laplacian(b["target"]) - laplacian(prediction)
        torch.testing.assert_close(first["global"], difference.flatten(1).norm(dim=1))


def test_spectral_identical_and_repeated_eigenvalues():
    real = torch.zeros(1, 5, 5)
    predicted = real.clone().requires_grad_()
    valid = torch.ones(1, 5, dtype=torch.bool)
    loss = spectral_loss(real, predicted, valid, 1., 0.)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    a = torch.tensor([[[0., 1.], [0., 0.]]])
    torch.testing.assert_close(laplacian(a), torch.tensor([[[.5, -.5], [-.5, .5]]]))


def test_contrast_equation_conflict_is_explicit():
    x = torch.zeros(2, 3)
    assert perturbation_loss(x, x, 1., "paper_separation") > .9
    assert perturbation_loss(x, x, 1., "invariance") < .01
    far = torch.full_like(x, 2.)
    assert perturbation_loss(x, far, 1., "paper_separation") < .01
    assert info_nce(torch.eye(4), torch.eye(4), .1) < .001


def test_dynamic_threshold_uses_only_prior_accepted_scores():
    threshold = CausalThreshold(10., .9, 4, 2)
    assert threshold.step(1.) == (0, 10.)
    assert threshold.step(2.) == (0, 10.)
    prediction, used = threshold.step(999.)
    assert prediction == 1 and abs(used-1.9) < 1e-6
    assert list(threshold.history) == [1., 2.]
