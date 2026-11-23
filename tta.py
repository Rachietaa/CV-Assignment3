"""Step 3: test-time adaptation on top of the best one-shot classifier, per model, over the 5 shot draws.

Two ways of using the test images at prediction time (labels are never used, DECISIONS.md D1):
  flip       average each test embedding with the embedding of its mirror image (test-time augmentation)
  transduct  EM-style refinement of the class weights on all unlabelled test embeddings

Usage:
    python tta.py                 # prompt set "templates+desc"
Writes results/tta_<prompts>.json.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from src import methods


def accuracy(feats, weights, labels):
    return 100 * (feats @ weights.T).argmax(dim=1).eq(labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', default='templates+desc')
    args = parser.parse_args()

    results = {}
    for model_dir in sorted(p for p in Path('features').iterdir() if (p / 'text.pt').exists()):
        test = torch.load(model_dir / 'test.pt')
        text = torch.load(model_dir / 'text.pt')[args.prompts]
        plain, flip = test['feats'], F.normalize(test['feats'] + test['feats_flip'], dim=-1)
        labels = test['labels']
        rows = {}
        for draw in sorted(model_dir.glob('shots_seed*.pt')):
            shots = torch.load(draw)
            torch.manual_seed(0)
            lp = methods.linear_probe_weights(text, shots['feats'], shots['views'])
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
        results[model_dir.name] = rows

        print(f'\n[{model_dir.name}]  prompts: {args.prompts}')
        for k, a in rows.items():
            spread = f'± {statistics.stdev(a):.1f}' if len(a) > 1 else ''
            print(f'  {k:34s} {statistics.mean(a):6.2f} {spread:7s}  [{", ".join(f"{x:.1f}" for x in a)}]')

    Path('results').mkdir(exist_ok=True)
    Path(f'results/tta_{args.prompts}.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
