"""Compute and cache all embeddings for one or more foundation models (frozen, forward passes only).

Usage:
    python extract_features.py --models clip-l14-336 dfn5b-h14-378
    python extract_features.py --models all

Writes features/<model>/:
    test.pt          test image embeddings [400, D], their flipped versions, labels
    shots_seed<k>.pt embeddings of the 16 shots of draw k [16, D] and of 32 random augmented views [16, 32, D]
    text.pt          one class embedding per prompt set (see src/prompts.py) [16, D]
Only the shot images listed in shots/ and the test images are opened (DECISIONS.md D1).
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import ImageOps

from src.data import class_names, load_shots, test_set
from src.fm import MODELS, augment, encode_class_prompts, encode_images, get_device, load
from src.prompts import prompt_sets

N_VIEWS = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help=f'any of {list(MODELS)} or "all"')
    parser.add_argument('--root', default='data')
    args = parser.parse_args()
    keys = list(MODELS) if args.models == ['all'] else args.models

    device = get_device()
    classes = class_names(args.root)
    test_paths, test_labels = test_set(args.root, classes)
    shot_files = sorted(Path('shots').glob('seed*.json'))
    print(f'Device {device} | {len(test_paths)} test images | {len(shot_files)} shot draws', flush=True)

    for key in keys:
        start = time.time()
        out = Path('features') / key
        out.mkdir(parents=True, exist_ok=True)
        model, preprocess, tokenizer = load(key, device)
        print(f'\n[{key}] loaded in {time.time() - start:.0f}s', flush=True)

        torch.save({'feats': encode_images(model, preprocess, args.root, test_paths, device),
                    'feats_flip': encode_images(model, preprocess, args.root, test_paths, device,
                                                view=ImageOps.mirror),
                    'labels': torch.tensor(test_labels), 'paths': test_paths, 'classes': classes},
                   out / 'test.pt')

        for f in shot_files:
            shot_classes, shot_paths = load_shots(f)
            assert shot_classes == classes
            torch.manual_seed(0)
            views = torch.stack([encode_images(model, preprocess, args.root, shot_paths, device, view=augment())
                                 for _ in range(N_VIEWS)], dim=1)
            torch.save({'feats': encode_images(model, preprocess, args.root, shot_paths, device),
                        'views': views, 'paths': shot_paths}, out / f'shots_{f.stem}.pt')

        sets = prompt_sets(classes)
        torch.save({name: encode_class_prompts(model, tokenizer, [p[c] for c in classes], device)
                    for name, p in sets.items()}, out / 'text.pt')
        print(f'[{key}] done in {time.time() - start:.0f}s -> {out}/', flush=True)
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
