"""One-shot classifiers on frozen foundation-model embeddings. All inputs are L2-normalized.

text   [C, D]     class embeddings from prompts (zero-shot classifier)
shot   [C, D]     embedding of the single labelled image of each class
views  [C, V, D]  embeddings of V random augmented views of that image
feats  [N, D]     images to classify
Each function returns logits [N, C].
"""
import torch
import torch.nn.functional as F


def zero_shot(feats, text):
    return feats @ text.T


def image_prototype(feats, shot, views=None):
    """Nearest class prototype; with views, the prototype is the mean of the image and its augmented views."""
    proto = shot if views is None else F.normalize(shot + views.sum(dim=1), dim=-1)
    return feats @ proto.T


def fused_prototype(feats, text, shot, views, alpha=0.5):
    """Classifier weight per class = normalized mix of its text embedding and its image prototype.
    alpha = weight of the text side (1.0 = zero-shot, 0.0 = image only)."""
    proto = F.normalize(shot + views.sum(dim=1), dim=-1)
    return feats @ F.normalize(alpha * text + (1 - alpha) * proto, dim=-1).T


def tip_adapter(feats, text, shot, views, alpha=1.0, beta=5.5):
    """Tip-Adapter (Zhang et al., 2022), training-free: zero-shot logits (scaled by CLIP's 100) plus a cache
    model whose keys are the embeddings of the shot and its views and whose values are their one-hot
    labels. The cache is averaged over each class's V+1 keys, so its scale is the same as the paper's
    1-shot setting (one key per class) and the paper's defaults alpha=1, beta=5.5 apply."""
    c, v, d = views.shape
    keys = torch.cat([shot[:, None], views], dim=1).reshape(-1, d)               # [C*(V+1), D]
    values = torch.arange(c).repeat_interleave(v + 1)
    affinity = torch.exp(-beta * (1 - feats @ keys.T))                            # [N, C*(V+1)]
    cache = affinity @ F.one_hot(values, c).float() / (v + 1)                     # [N, C], in [0, 1]
    return 100 * feats @ text.T + alpha * cache


def sinkhorn_balance(probs, iters=3):
    """Rescale soft assignments [N, C] so that every class receives the same total mass (N / C) while each
    image still distributes a total of 1 (Sinkhorn-Knopp). Encodes a uniform class prior."""
    n, c = probs.shape
    p = probs
    for _ in range(iters):
        p = p / p.sum(dim=0, keepdim=True).clamp_min(1e-12) * (n / c)   # equal class totals
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)             # each image sums to 1
    return p


def transductive_refine(feats, weights, iters=10, temperature=0.01, anchor=0.5, balanced=False):
    """Test-time adaptation without labels: EM-style refinement of the class weights on the unlabelled
    test embeddings. Each iteration soft-assigns every test image to the classes (softmax of cosine
    similarity / temperature), recomputes each class centre as the probability-weighted mean of the test
    embeddings, and mixes it with the original weights (`anchor`) so classes cannot drift away from what
    the text and the shot say. Uses no test labels. balanced=True adds a uniform class prior: the soft
    assignments are rebalanced (sinkhorn_balance) so no class can absorb another class's images."""
    w = weights.clone()
    for _ in range(iters):
        probs = (feats @ w.T / temperature).softmax(dim=1)                 # [N, C]
        if balanced:
            probs = sinkhorn_balance(probs)
        centres = F.normalize(probs.T @ feats, dim=-1)                     # [C, D]
        w = F.normalize(anchor * weights + (1 - anchor) * centres, dim=-1)
    return w


def linear_probe_weights(text, shot, views, steps=300, lr=1e-3, reg=1.0, temperature=0.01):
    """Weights of the text-initialized linear probe (see linear_probe)."""
    c, v, d = views.shape
    x = torch.cat([shot[:, None], views], dim=1).reshape(-1, d)
    y = torch.arange(c).repeat_interleave(v + 1)
    w = text.clone().requires_grad_(True)
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=0.0)
    for _ in range(steps):
        logits = x @ F.normalize(w, dim=-1).T / temperature
        loss = F.cross_entropy(logits, y) + reg * (w - text).pow(2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return F.normalize(w.detach(), dim=-1)


def linear_probe(feats, text, shot, views, **kwargs):
    """A linear classifier trained on the 16 shots and their augmented views, initialized from the text
    embeddings and pulled back towards them by an L2 penalty (`reg`), so one image per class cannot drag
    the weights far from what the language side already knows."""
    return feats @ linear_probe_weights(text, shot, views, **kwargs).T
