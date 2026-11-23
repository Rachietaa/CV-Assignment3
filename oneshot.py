"""Step 2: one-shot methods on every cached model, over the 5 random shot draws (mean ± std).

Usage:
    python oneshot.py                          # prompt set "templates+desc"
    python oneshot.py --prompts templates
    python oneshot.py --sweep                  # also print the text/image mixing curve (analysis only)

Writes results/oneshot_<prompts>.json.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from src import methods

METHODS = {
    'zero-shot (text only)': lambda t, s, v, f: methods.zero_shot(f, t),
    'image prototype (shot only)': lambda t, s, v, f: methods.image_prototype(f, s),
    'image prototype (shot + 32 views)': lambda t, s, v, f: methods.image_prototype(f, s, v),
    'text + image prototype (alpha 0.5)': lambda t, s, v, f: methods.fused_prototype(f, t, s, v, 0.5),
    'Tip-Adapter (alpha 1, beta 5.5)': lambda t, s, v, f: methods.tip_adapter(f, t, s, v),
    'linear probe, text-initialized': lambda t, s, v, f: methods.linear_probe(f, t, s, v),
}


def accuracy(logits, labels):
    return 100 * logits.argmax(dim=1).eq(labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', default='templates+desc')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    results = {}
    for model_dir in sorted(p for p in Path('features').iterdir() if (p / 'text.pt').exists()):
        test = torch.load(model_dir / 'test.pt')
        text = torch.load(model_dir / 'text.pt')[args.prompts]
        draws = sorted(model_dir.glob('shots_seed*.pt'))
        torch.manual_seed(0)
        accs = {m: [] for m in METHODS}
        for draw in draws:
            shots = torch.load(draw)
            for name, fn in METHODS.items():
                accs[name].append(accuracy(fn(text, shots['feats'], shots['views'], test['feats']), test['labels']))
        results[model_dir.name] = accs

        print(f'\n[{model_dir.name}]  prompts: {args.prompts}  ({len(draws)} shot draws)')
        for name, a in accs.items():
            spread = f'± {statistics.stdev(a):.1f}' if len(a) > 1 else ''
            print(f'  {name:36s} {statistics.mean(a):6.2f} {spread:7s}  [{", ".join(f"{x:.1f}" for x in a)}]')

        if args.sweep:
            shots = torch.load(draws[0])
            curve = [(a / 10, accuracy(methods.fused_prototype(test['feats'], text, shots['feats'], shots['views'],
                                                               a / 10), test['labels'])) for a in range(11)]
            print('  text weight alpha -> test acc (draw 0): ' + ', '.join(f'{a:.1f}:{acc:.1f}' for a, acc in curve))

    Path('results').mkdir(exist_ok=True)
    Path(f'results/oneshot_{args.prompts}.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
