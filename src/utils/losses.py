import torch.nn.functional as F


def cosine_loss(a, b):
    """Negative cosine similarity loss (minimize for alignment)."""
    return 1 - F.cosine_similarity(a, b).mean()