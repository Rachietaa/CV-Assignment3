# ITCS 6169/8169 — Assignment 3: One-Shot Scene Classification

16-class scene recognition using **only one labelled training image per class** (16 images in total), with
five frozen foundation models (CLIP, DFN, Perception Encoder, SigLIP, SigLIP 2).

## Final result

| | Test accuracy (400 images) |
|---|---:|
| **Final model (shot draw 0, fixed in advance)** | **94.25%** |
| Same pipeline over 5 random one-shot draws | 95.7 ± 1.4% (94.25–97.25%) |
| Zero-shot, no labelled image at all (best single model) | 91.25% |
| *Assignment 1: fine-tuned ImageNet ensemble, 1,920 labelled images* | *95.75%* |

**Final model:** for each of five frozen foundation models, a linear classifier that starts from the class
*text* embeddings and is trained on the 16 shot images (plus 32 augmented views of each), with a penalty that
keeps it close to the text. The five models are combined in one joint embedding space, and at test time the
class weights are adapted to the unlabelled test images (transductive refinement with a uniform class
prior), with flip averaging.

The full experimental history is in [EXPERIMENTS.md](EXPERIMENTS.md), every design decision and its reason in
[DECISIONS.md](DECISIONS.md), and AI tool usage in [AI_USAGE.md](AI_USAGE.md).

## Rules followed

- **Exactly one labelled training image per class.** The 16 images are chosen at random with a fixed seed
  ([shots/seed0.json](shots/seed0.json)); the code never opens any other training image
  ([src/data.py](src/data.py)).
- **Test labels are used only to compute accuracy.** Test-time adaptation uses the unlabelled test images.
- No labelled validation set exists under these rules, so method settings are paper defaults or fixed in
  advance, and the final model is the ensemble of all five models, not the best-scoring one.

## Main results (test accuracy, mean ± std over 5 shot draws)

| Method | Best single model | 5-model ensemble |
|---|---:|---:|
| Zero-shot (text only, 0 images) | 91.25 (CLIP L/14) | 90.00 |
| Zero-shot + transductive refinement (0 images) | 93.75 (CLIP L/14) | 93.50 |
| Image prototype from the single shot | 79.1 ± 1.5 (DFN-5B) | — |
| Text-initialized linear probe (16 images) | 93.55 ± 1.4 (CLIP L/14) | 92.9 ± 0.8 |
| + flip + transductive refinement | 95.35 ± 0.5 (SigLIP 2, without flip) | 95.0 ± 1.4 |
| **+ class-balance prior in the refinement** | 96.05 ± 0.6 (CLIP L/14) | **95.7 ± 1.4** |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The foundation-model weights (~16 GB in total) are downloaded automatically from Hugging Face by
`open_clip` on first use. Hardware used: MacBook (Apple M5, 32 GB), PyTorch on the Apple GPU (MPS); the device
is chosen automatically (CUDA, then MPS, then CPU).

| Key | open_clip model | Pretrained weights |
|---|---|---|
| `clip-l14-336` | ViT-L-14-336 (QuickGELU) | `openai` |
| `dfn5b-h14-378` | ViT-H-14-378-quickgelu | `dfn5b` |
| `pe-core-l14-336` | PE-Core-L-14-336 | `meta` |
| `siglip-so400m-384` | ViT-SO400M-14-SigLIP-384 | `webli` |
| `siglip2-so400m-384` | ViT-SO400M-16-SigLIP2-384 | `webli` |

## Data

Same dataset as Assignments 1 and 2. Unpack it (or symlink it) so that:

```text
data/
  train/<class_name>/*.jpg   2,400 images (ONLY the 16 listed in shots/seed0.json are used)
  test/<class_name>/*.jpg    400 images, 25 per class
  test2/*.jpg                400 unlabeled images (not used)
```

## Reproducing the final result

**Evaluate the trained classifier** ([final/classifier_seed0.pt](final/classifier_seed0.pt), 0.7 MB, included
in the repository) directly on the raw test images:

```bash
python predict.py --classifier final/classifier_seed0.pt --images data/test --out predictions_test.csv
```

It encodes the images with the five models, adapts to them without labels, writes one prediction per image
and prints the accuracy (94.25%). Because of the test-time adaptation, images are classified together as
a set; the same command works on any folder of images, e.g. `--images data/test2`.

**Rebuild the classifier from scratch:**

```bash
python select_shots.py --seeds 0 1 2 3 4               # the random one-shot draws (already in shots/)
python extract_features.py --models all                # embeddings of the 16 shots per draw + test images
python train_final.py --seed 0                         # -> final/classifier_seed0.pt
```

**Reproduce every table in EXPERIMENTS.md** (from the cached embeddings, seconds each):

```bash
python zeroshot.py     # Step 1: zero-shot, 4 prompt sets
python oneshot.py      # Step 2: one-shot methods, 5 draws
python tta.py          # Step 3: test-time adaptation
python ensemble.py     # Step 4: 5-model joint ensemble
```

## Repository layout

```text
src/data.py          one-shot data access (only the listed shots and the test images are opened), shot selection
src/fm.py            the five foundation models (open_clip): loading, image and prompt embeddings
src/prompts.py       class names, prompt templates and short class descriptions
src/methods.py       zero-shot, image prototypes, text-image fusion, Tip-Adapter, text-initialized
                     linear probe, transductive refinement
select_shots.py      random one-image-per-class draws -> shots/seed<k>.json
extract_features.py  embedding cache -> features/<model>/ (not in git; reproducible)
zeroshot.py, oneshot.py, tta.py, ensemble.py   experiment steps 1-4 -> results/
train_final.py       builds the final classifier -> final/classifier_seed0.pt
predict.py           inference on any folder of images -> CSV (+ accuracy if labelled)
```
