# ITCS 6169/8169 — Assignment 3: One-Shot Scene Classification

16-class scene recognition using **only one labelled training image per class** (16 images in total), with
frozen foundation models (CLIP / SigLIP family).

*(Results, recipe and reproduction steps will be added as experiments are completed.)*

## Data

Same dataset as Assignments 1 and 2. Unpack it (or symlink it) so that:

```text
data/
  train/<class_name>/*.jpg   2,400 images, 150 per class (ONLY the 16 listed in shots/seed<k>.json are used)
  test/<class_name>/*.jpg    400 images, 25 per class
  test2/*.jpg                400 unlabeled images (not used)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
