"""Keep the legacy reconstruction separate from the inductive extension."""
from .model import MTGGAN


def make_model(cfg):
    if cfg.model.architecture == "legacy":
        return MTGGAN(cfg)
    from .inductive_model import InductiveMTGGAN
    return InductiveMTGGAN(cfg)


def generate(model, batch, **kwargs):
    extra = {"present": batch["node_present"]} if hasattr(model, "inductive") else {}
    return model.generator(batch["features"], batch["adjacency"], batch["valid"], **extra, **kwargs)


def discriminate(model, batch, adjacency, *, observed=False, present=None):
    if not hasattr(model, "inductive"):
        return model.discriminator(adjacency)
    return model.discriminator(adjacency, batch["features"], batch["valid"],
                               present=present if present is not None else batch["target_present"] if observed else None)
