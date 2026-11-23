"""Step 4: combining the foundation models, over the 5 shot draws.

Joint space: each model's L2-normalized embedding is divided by sqrt(M) and the M vectors are concatenated,
so the joint vector has norm 1 and its cosine similarity with a joint class weight equals the average of the
M per-model cosine similarities. Transductive refinement can then run once, in that joint space.

Usage:
    python ensemble.py
    python ensemble.py --models clip-l14-336 siglip2-so400m-384
Writes results/ensemble.json.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from src import methods


def joint(parts):
    return torch.cat([p / len(parts) ** 0.5 for p in parts], dim=-1)


def accuracy(feats, weights, labels):
    return 100 * (feats @ weights.T).argmax(dim=1).eq(labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None, help='default: every model in features/')
    parser.add_argument('--prompts', default='templates+desc')
    args = parser.parse_args()
    keys = args.models or sorted(p.name for p in Path('features').iterdir() if (p / 'text.pt').exists())

    tests = {k: torch.load(Path('features') / k / 'test.pt') for k in keys}
    texts = {k: torch.load(Path('features') / k / 'text.pt')[args.prompts] for k in keys}
    labels = tests[keys[0]]['labels']
    plain = joint([tests[k]['feats'] for k in keys])
    flip = joint([F.normalize(tests[k]['feats'] + tests[k]['feats_flip'], dim=-1) for k in keys])
    text = joint([texts[k] for k in keys])

    rows = {}
    for seed in range(5):
        torch.manual_seed(0)
        lp = joint([methods.linear_probe_weights(texts[k], *[torch.load(Path('features') / k / f'shots_seed{seed}.pt')[x]
                                                             for x in ('feats', 'views')]) for k in keys])
        variants = {
            'zero-shot': accuracy(plain, text, labels),
            'zero-shot + transduct': accuracy(plain, methods.transductive_refine(plain, text), labels),
            'linear probe': accuracy(plain, lp, labels),
            'linear probe + flip': accuracy(flip, lp, labels),
            'linear probe + transduct': accuracy(plain, methods.transductive_refine(plain, lp), labels),
            'linear probe + flip + transduct': accuracy(flip, methods.transductive_refine(flip, lp), labels),
        }
        for k, v in variants.items():
            rows.setdefault(k, []).append(v)

    print(f'Ensemble of {len(keys)} models: {", ".join(keys)}  (prompts: {args.prompts})')
    for k, a in rows.items():
        print(f'  {k:34s} {statistics.mean(a):6.2f} ± {statistics.stdev(a):.1f}  [{", ".join(f"{x:.1f}" for x in a)}]')
    Path('results').mkdir(exist_ok=True)
    name = 'all' if args.models is None else '+'.join(keys)
    out = Path('results/ensemble.json')
    saved = json.loads(out.read_text()) if out.exists() else {}
    saved[name] = rows
    out.write_text(json.dumps(saved, indent=2))


if __name__ == '__main__':
    main()
