"""Report figure from the saved results (no model is run).

Usage:
    python make_figures.py      # -> figures/results.png
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Categorical palette validated for colour-vision deficiency; text stays in neutral ink.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
MODELS = {'clip-l14-336': 'CLIP', 'dfn5b-h14-378': 'DFN-5B', 'pe-core-l14-336': 'PE-Core',
          'siglip-so400m-384': 'SigLIP', 'siglip2-so400m-384': 'SigLIP 2'}


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)


def main():
    ens = json.loads(Path('results/ensemble.json').read_text())['all']
    one = json.loads(Path('results/oneshot_templates+desc.json').read_text())
    tta = json.loads(Path('results/tta_templates+desc.json').read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw={'width_ratios': [1.0, 1.0]})

    # Left: the accuracy ladder of the 5-model ensemble; dots are the 5 shot draws.
    steps = [('zero-shot (0 images)', 'zero-shot'),
             ('+ balanced refinement (0 images)', 'zero-shot + transduct (balanced)'),
             ('text-initialized probe (16 images)', 'linear probe'),
             ('+ flip + refinement', 'linear probe + flip + transduct'),
             ('+ class-balance prior (final)', 'linear probe + flip + transduct (balanced)')]
    for i, (label, key) in enumerate(steps):
        vals = ens[key]
        y = len(steps) - 1 - i
        ax1.barh(y, statistics.mean(vals) - 85, left=85, height=0.55,
                 color=BLUE if i == len(steps) - 1 else '#9cc0ea')
        ax1.scatter(vals, [y] * len(vals), s=10, color=INK, zorder=3)
        ax1.text(max(vals) + 0.25, y, f'{statistics.mean(vals):.1f}', va='center', fontsize=7.5, color=INK)
    ax1.axvline(95.75, color=MUTED, linestyle='--', linewidth=1)
    ax1.text(95.85, len(steps) - 0.45, 'A1, 1,920 labels', fontsize=7, color=MUTED, va='bottom')
    ax1.set_xlim(85, 99)
    ax1.set_ylim(-0.6, len(steps) - 0.1)
    ax1.set_yticks(range(len(steps)))
    ax1.set_yticklabels([label for label, _ in steps][::-1], fontsize=7.5, color=INK)
    ax1.tick_params(axis='y', length=0)
    ax1.set_xlabel('test accuracy (%), 5-model ensemble', fontsize=8, color=INK)
    style(ax1)

    # Right: per model, one image vs the text vs the full pipeline.
    keys = list(MODELS)
    x = range(len(keys))
    series = [('one image', [statistics.mean(one[k]['image prototype (shot + 32 views)']) for k in keys], ORANGE),
              ('text only', [statistics.mean(one[k]['zero-shot (text only)']) for k in keys], AQUA),
              ('full pipeline', [statistics.mean(tta[k]['linear probe + flip + transduct (balanced)']) for k in keys], BLUE)]
    width = 0.26
    for j, (label, vals, color) in enumerate(series):
        ax2.bar([i + (j - 1) * width for i in x], vals, width=width - 0.03, color=color, label=label)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([MODELS[k] for k in keys], fontsize=7, rotation=0)
    ax2.set_ylim(60, 100)
    ax2.set_ylabel('test accuracy (%)', fontsize=8, color=INK)
    ax2.grid(axis='y', color=GRID, linewidth=0.8)
    ax2.legend(fontsize=7, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1)
    style(ax2)

    fig.tight_layout()
    Path('figures').mkdir(exist_ok=True)
    fig.savefig('figures/results.png', dpi=220)
    print('Saved figures/results.png')


if __name__ == '__main__':
    main()
