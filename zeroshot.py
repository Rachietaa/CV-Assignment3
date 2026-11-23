"""Step 1: zero-shot classification (no training images at all) for every cached model and prompt set.

Usage:
    python zeroshot.py            # all models in features/

Each class is represented only by its text embedding; an image is assigned to the class with the highest
cosine similarity. Accuracy is also reported on the 16 seed-0 shots: zero-shot methods never train on
them, so they serve as a (tiny) legal check set for choosing settings (DECISIONS.md D3).
Writes results/zeroshot.json.
"""
import json
from pathlib import Path

import torch


def accuracy(feats, labels, weights):
    return (feats @ weights.T).argmax(dim=1).eq(labels).float().mean().item()


def main():
    rows = []
    for model_dir in sorted(p for p in Path('features').iterdir() if (p / 'text.pt').exists()):
        test = torch.load(model_dir / 'test.pt')
        shots = torch.load(model_dir / 'shots_seed0.pt')
        text = torch.load(model_dir / 'text.pt')
        shot_labels = torch.arange(len(test['classes']))
        for name, weights in text.items():
            flip_avg = torch.nn.functional.normalize(test['feats'] + test['feats_flip'], dim=-1)
            rows.append({'model': model_dir.name, 'prompts': name,
                         'shots_acc': accuracy(shots['feats'], shot_labels, weights),
                         'test_acc': accuracy(test['feats'], test['labels'], weights),
                         'test_acc_flip': accuracy(flip_avg, test['labels'], weights)})

    print(f"{'model':22s} {'prompts':16s} {'16 shots':>9s} {'test':>7s} {'test+flip':>10s}")
    for r in rows:
        print(f"{r['model']:22s} {r['prompts']:16s} {100 * r['shots_acc']:8.1f}% {100 * r['test_acc']:6.2f}% "
              f"{100 * r['test_acc_flip']:9.2f}%")
    Path('results').mkdir(exist_ok=True)
    Path('results/zeroshot.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
