"""Choose the one labelled training image per class (DECISIONS.md D2) and save the list.

Usage:
    python select_shots.py --seeds 0 1 2 3 4     # -> shots/seed0.json ... shots/seed4.json
Draw 0 is the main draw; the others measure how much the choice of the single image matters.
"""
import argparse
import json
from pathlib import Path

from src.data import select_shots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    out = Path('shots')
    out.mkdir(exist_ok=True)
    for seed in args.seeds:
        shots = select_shots(args.root, seed)
        (out / f'seed{seed}.json').write_text(json.dumps(shots, indent=2) + '\n')
        print(f'seed {seed}: {len(shots)} images -> shots/seed{seed}.json')


if __name__ == '__main__':
    main()
