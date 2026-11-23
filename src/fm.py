"""Frozen foundation models (open_clip): loading and L2-normalized image / text embeddings."""
import torch
import torch.nn.functional as F
from torchvision import transforms

from .data import load_rgb

# key -> (open_clip architecture, pretrained tag)
MODELS = {
    'clip-l14-336': ('ViT-L-14-336', 'openai'),               # original OpenAI CLIP
    'dfn5b-h14-378': ('ViT-H-14-378-quickgelu', 'dfn5b'),    # Apple DFN, data-filtered CLIP
    'siglip-so400m-384': ('ViT-SO400M-14-SigLIP-384', 'webli'),
    'siglip2-so400m-384': ('ViT-SO400M-16-SigLIP2-384', 'webli'),
    'pe-core-l14-336': ('PE-Core-L-14-336', 'meta'),         # Meta Perception Encoder
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load(key, device):
    import open_clip
    arch, tag = MODELS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
    tokenizer = open_clip.get_tokenizer(arch)
    return model.eval().to(device), preprocess, tokenizer


def augment(size=256, scale=(0.5, 1.0)):
    """Random views of a shot image (applied before the model's own resize/crop/normalize)."""
    return transforms.Compose([transforms.RandomResizedCrop(size, scale=scale),
                               transforms.RandomHorizontalFlip()])


@torch.inference_mode()
def encode_images(model, preprocess, root, paths, device, view=None, batch_size=32):
    """L2-normalized image embeddings [N, D]. view: optional PIL -> PIL transform applied first."""
    feats = []
    for i in range(0, len(paths), batch_size):
        imgs = [load_rgb(f'{root}/{p}') for p in paths[i:i + batch_size]]
        if view is not None:
            imgs = [view(im) for im in imgs]
        x = torch.stack([preprocess(im) for im in imgs]).to(device)
        feats.append(F.normalize(model.encode_image(x).float(), dim=-1).cpu())
    return torch.cat(feats)


@torch.inference_mode()
def encode_class_prompts(model, tokenizer, prompts_per_class, device):
    """One embedding per class: the normalized mean of its normalized prompt embeddings. [C, D]"""
    out = []
    for prompts in prompts_per_class:
        t = F.normalize(model.encode_text(tokenizer(prompts).to(device)).float(), dim=-1)
        out.append(F.normalize(t.mean(dim=0), dim=-1).cpu())
    return torch.stack(out)
