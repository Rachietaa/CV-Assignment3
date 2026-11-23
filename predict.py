"""Inference with the final one-shot classifier.

Encodes the images with the five frozen foundation models (and their mirror images), combines them in the
joint space, adapts the class weights to the batch with transductive refinement (no labels), and predicts.
Because of the refinement, images should be classified together as a set (e.g. a whole test folder).

Usage:
    python predict.py --classifier final/classifier_seed0.pt --images data/test
    python predict.py --classifier final/classifier_seed0.pt --images data/test2 --out predictions_test2.csv

If the folder has one sub-folder per class (like data/test), accuracy is also printed.
Writes a CSV: image, predicted_class.
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import ImageOps

from src import methods
from src.data import EXTENSIONS
from src.fm import encode_images, get_device, load


def joint(parts):
    return torch.cat([p / len(parts) ** 0.5 for p in parts], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', default='final/classifier_seed0.pt')
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', default='predictions.csv')
    args = parser.parse_args()

    clf = torch.load(args.classifier)
    classes = clf['classes']
    root = Path(args.images)
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in EXTENSIONS)
    rel = [str(p.relative_to(root)) for p in paths]
    labelled = all(p.parent.name in classes for p in paths)
    device = get_device()

    feats, weights = [], []
    for key, m in clf['models'].items():
        model, preprocess, _ = load(key, device)
        plain = encode_images(model, preprocess, root, rel, device)
        flip = encode_images(model, preprocess, root, rel, device, view=ImageOps.mirror)
        feats.append(F.normalize(plain + flip, dim=-1) if clf['flip'] else plain)
        weights.append(m['weights'])
        print(f'{key}: encoded {len(rel)} images', flush=True)
        del model

    x, w = joint(feats), joint(weights)
    w = methods.transductive_refine(x, w, **clf['transductive'])
    pred = (x @ w.T).argmax(dim=1)

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'predicted_class'])
        writer.writerows((r, classes[i]) for r, i in zip(rel, pred.tolist()))
    print(f'Wrote {len(rel)} predictions to {args.out}')

    if labelled:
        y = torch.tensor([classes.index(p.parent.name) for p in paths])
        print(f'Accuracy: {100 * pred.eq(y).float().mean():.2f}% ({int(pred.eq(y).sum())}/{len(y)})')


if __name__ == '__main__':
    main()
