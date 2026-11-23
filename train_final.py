"""Build the final one-shot classifier (DECISIONS.md D6) from the 16 shots of one draw and save it.

For each of the five foundation models: the class text embeddings and the text-initialized linear-probe
weights trained on the 16 shots + 32 augmented views each. Uses the cached shot / text embeddings written
by extract_features.py (run that first). Nothing from the test set is used here.

Usage:
    python train_final.py --seed 0          # -> final/classifier_seed0.pt  (draw 0 is the main draw, D2)
"""
import argparse
from pathlib import Path

import torch

from src import methods
from src.fm import MODELS

FINAL_MODELS = ['clip-l14-336', 'dfn5b-h14-378', 'pe-core-l14-336', 'siglip-so400m-384', 'siglip2-so400m-384']
PROMPTS = 'templates+desc'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    shots_file = Path('shots') / f'seed{args.seed}.json'
    per_model = {}
    for key in FINAL_MODELS:
        shots = torch.load(Path('features') / key / f'shots_seed{args.seed}.pt')
        text = torch.load(Path('features') / key / 'text.pt')[PROMPTS]
        torch.manual_seed(0)
        per_model[key] = {'arch': MODELS[key][0], 'pretrained': MODELS[key][1], 'text': text,
                          'weights': methods.linear_probe_weights(text, shots['feats'], shots['views'])}
        print(f'{key}: linear probe trained on {len(shots["paths"])} shots')

    classes = torch.load(Path('features') / FINAL_MODELS[0] / 'test.pt')['classes']
    out = Path('final')
    out.mkdir(exist_ok=True)
    torch.save({'classes': classes, 'models': per_model, 'prompts': PROMPTS, 'shots': shots_file.read_text(),
                'transductive': {'iters': 10, 'temperature': 0.01, 'anchor': 0.5, 'balanced': True}, 'flip': True},
               out / f'classifier_seed{args.seed}.pt')
    print(f'Saved {out}/classifier_seed{args.seed}.pt')


if __name__ == '__main__':
    main()
