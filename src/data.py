"""Data access under the one-shot rule.

Only two kinds of images are ever opened:
  * the 16 training images listed in shots/seed<k>.json (one per class), and
  * the test images (labels used only to compute accuracy, never by a method).
No other training image is read.
"""
import json
import random
from pathlib import Path

from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTENSIONS)


def class_names(root):
    return sorted(p.name for p in (Path(root) / 'train').iterdir() if p.is_dir())


def load_rgb(path):
    """Open an image as 3-channel RGB (most images are stored as grayscale; Flower images are colour)."""
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')


def select_shots(root, seed):
    """Pick one training image per class uniformly at random (fixed seed). Only file *names* are listed;
    no image other than the chosen one is opened."""
    rng = random.Random(seed)
    return {cls: str(rng.choice(list_images(Path(root) / 'train' / cls)).relative_to(root))
            for cls in class_names(root)}


def load_shots(shots_file):
    """Return (classes, paths) of a saved shot list, in class order."""
    shots = json.loads(Path(shots_file).read_text())
    classes = sorted(shots)
    return classes, [shots[c] for c in classes]


def test_set(root, classes):
    """All labelled test images as (paths, labels)."""
    paths, labels = [], []
    for i, cls in enumerate(classes):
        for p in list_images(Path(root) / 'test' / cls):
            paths.append(str(p.relative_to(root)))
            labels.append(i)
    return paths, labels
