# Experiment Log

Test accuracy on the 400 labelled test images (25 per class). Methods that use the 16 shots are run on
5 random shot draws (seeds 0–4) and reported as mean ± std. There is **no labelled validation set** (only
one training image per class may be used), so every comparison below is on test; to limit test-set
selection, all method settings are paper defaults or fixed in advance (DECISIONS.md D3), and the final
model was not chosen as the best-scoring single model (D6).

## Summary

| Step | Method | Labelled images | Best single model | 5-model ensemble |
|---|---|---:|---:|---:|
| 1 | Zero-shot, best prompts | 0 | 91.25 (CLIP L/14) | 90.00 |
| 1+3 | Zero-shot + transductive refinement | 0 | 93.75 (CLIP L/14) | 93.50 |
| 2 | Text-initialized linear probe | 16 | 93.55 ± 1.4 (CLIP L/14) | 92.9 ± 0.8 |
| 2+3 | Linear probe + flip + transductive refinement | 16 | 95.35 ± 0.5 (SigLIP 2) | 95.0 ± 1.4 |
| **5** | **+ class-balance prior in the refinement (D8)** | **16** | **96.05 ± 0.6** (CLIP L/14) | **95.7 ± 1.4** ← final; draw 0 = 94.25 |
| *A1* | *Fine-tuned ImageNet ConvNeXt V2 ensemble* | *1,920* | | *95.75* |

---

## Step 1 — Zero-shot classification (no images)

Each class is represented only by a text embedding; an image goes to the most similar class.
`python extract_features.py --models all` then `python zeroshot.py` (results/zeroshot.json).

| Model | Raw folder names | Natural names | + 8 templates | + 2 descriptions per class |
|---|---:|---:|---:|---:|
| OpenAI CLIP ViT-L/14 @336 | 86.25 | 87.75 | 88.50 | **91.25** |
| Apple DFN-5B ViT-H/14 @378 | 86.50 | 86.50 | 88.25 | 89.75 |
| Meta PE-Core L/14 @336 | 82.50 | 88.25 | 88.25 | 89.25 |
| Google SigLIP SO400M @384 | 80.00 | 88.25 | 88.00 | 88.75 |
| Google SigLIP 2 SO400M @384 | 89.25 | 88.75 | 89.00 | 89.50 |

- **Every model reaches ~89–91% with no labelled image at all.** These models already know what a bedroom
  or a coast looks like.
- **Prompt engineering matters, most for SigLIP**: "a photo of a InsideCity" → "a photo of a city centre"
  is +8.25 points; templates that say "black and white photo" plus short scene descriptions add 1–3 more
  for most models.
- **Bigger or newer is not better here**: the oldest model (OpenAI CLIP, 2021) is the best zero-shot model;
  the largest (DFN-5B ViT-H) is second.
- The OpenAI CLIP numbers above use QuickGELU, the activation its weights were trained with. The first run
  used plain GELU by mistake and was 2.0 points lower zero-shot (89.25).
- The 16 shots were meant as a tiny legal check set for choosing prompts (D3), but on 16 images one image is
  6.25 points and the check could not distinguish the prompt sets.

## Step 2 — One-shot methods (16 labelled images)

`python oneshot.py` (results/oneshot_templates+desc.json).

| Method | CLIP L/14 | DFN-5B | PE-Core | SigLIP | SigLIP 2 |
|---|---:|---:|---:|---:|---:|
| Zero-shot (text only, 0 images) | 91.25 | 89.75 | 89.25 | 88.75 | 89.50 |
| Image prototype (the shot only) | 68.9 ± 3.9 | 79.1 ± 1.5 | 72.6 ± 5.9 | 75.9 ± 3.6 | 72.4 ± 3.3 |
| Image prototype (shot + 32 augmented views) | 73.2 ± 4.3 | 82.7 ± 1.6 | 73.0 ± 6.0 | 78.2 ± 3.8 | 77.2 ± 3.7 |
| Text + image prototype (α = 0.5) | 87.3 ± 2.4 | 90.9 ± 2.2 | 86.1 ± 3.2 | 88.5 ± 2.1 | 87.9 ± 1.4 |
| Tip-Adapter (paper defaults α 1, β 5.5) | 91.3 ± 0.2 | 90.2 ± 0.2 | 89.4 ± 0.1 | 88.9 ± 0.1 | 90.0 ± 0.2 |
| **Linear probe, initialized from text** | **93.6 ± 1.4** | **91.9 ± 1.8** | **91.4 ± 1.7** | **91.2 ± 1.0** | **92.3 ± 0.8** |

**Linear probe, initialized from text** (`src/methods.py`): the classifier weights start as the class text
embeddings and are trained (300 Adam steps) with cross-entropy on the 16 shots and their 32 augmented views
each, plus an L2 penalty pulling the weights back towards the text embeddings. The penalty is what makes one
image per class usable: the image can adjust the class direction, but not replace what the text says.

- **The text is worth far more than one image.** One image per class gives 69–79%; the class name gives
  89–91%. A single photo is one noisy example of its class; the text describes the whole category.
- **Which image is drawn matters a lot for image-only methods** (e.g. PE-Core 64.5–81.0% across draws).
- **Naively mixing text and image prototypes hurts** for most models (α = 0.5 is below zero-shot): an
  untuned 50/50 mix gives the noisy single image too much weight.
- **Tip-Adapter with its paper defaults barely moves** from zero-shot (±0.2): with one shot per class the
  cache term is small compared to the text logits.
- **The text-initialized linear probe is the best one-shot method for every model** (+2.1 to +2.8 over
  zero-shot on average). It is not guaranteed for every draw: on 3 models one draw falls slightly below
  zero-shot (e.g. CLIP draw 0: 91.0 vs 91.25), so the benefit of one image depends on which image it is.

## Step 3 — Test-time adaptation (unlabelled test images)

`python tta.py` (results/tta_templates+desc.json). Test labels are never used.

- **flip**: each test embedding is averaged with the embedding of its mirror image.
- **transductive refinement** (`methods.transductive_refine`): 10 iterations of soft-assigning every test
  image to the classes (softmax of cosine similarity / 0.01), recomputing each class centre as the
  probability-weighted mean of the test embeddings, and mixing it 50/50 with the starting weights. Settings
  fixed in advance, not tuned.

| | CLIP L/14 | DFN-5B | PE-Core | SigLIP | SigLIP 2 |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 91.25 | 89.75 | 89.25 | 88.75 | 89.50 |
| Zero-shot + transductive | 93.75 | 93.50 | 92.75 | 93.00 | 93.00 |
| Linear probe | 93.6 ± 1.4 | 91.9 ± 1.8 | 91.4 ± 1.7 | 91.2 ± 1.0 | 92.3 ± 0.8 |
| Linear probe + flip | 93.6 ± 1.2 | 92.3 ± 1.9 | 91.5 ± 1.9 | 91.2 ± 0.9 | 92.2 ± 1.1 |
| Linear probe + transductive | 95.0 ± 1.3 | 94.5 ± 0.5 | 93.4 ± 0.9 | 94.3 ± 4.0 | **95.35 ± 0.5** |
| Linear probe + flip + transductive | 95.25 ± 1.2 | 94.7 ± 0.9 | 93.25 ± 0.8 | 94.5 ± 4.2 | 95.0 ± 1.0 |

- **Using the unlabelled test images is worth +2.5 to +4.3 points even with zero labelled images**, and
  +1.4 to +3.1 on top of the linear probe. The refinement moves each class centre towards where that
  class's test images actually cluster, which a single shot or a text prompt can only approximate.
- **16 labelled images + transductive refinement ≈ 95%, about equal to Assignment 1's fine-tuned ImageNet
  ensemble trained on 1,920 labelled images (95.75%).**
- **Risk of transduction:** for SigLIP on draw 0 the probe + refinement *dropped* to 87.5% (other draws
  93.8–97.3). If the starting classifier is wrong for a class, the refinement can pull that class towards
  another class's cluster. This is the large ± 4.0 for SigLIP.
- Flip test-time augmentation adds little (≤ 0.4), as in A1.

## Step 4 — Combining the five models

`python ensemble.py` (results/ensemble.json). Each model's normalized embedding is divided by √5 and the five
are concatenated: in this joint space, cosine similarity equals the average of the five models' cosine
similarities, so the linear probes and the transductive refinement run unchanged, once, using all models.

| 5-model joint space | Test acc |
|---|---:|
| Zero-shot | 90.00 |
| Zero-shot + transductive | 93.50 |
| Linear probe | 92.9 ± 0.8 |
| Linear probe + flip | 92.95 ± 0.8 |
| Linear probe + transductive | 94.95 ± 1.5 |
| Linear probe + flip + transductive (final model before D8) | 95.0 ± 1.4 |

- **The ensemble does not beat the best single model** (95.0 vs 95.35). Ensembles help when models make
  different mistakes; here the mistakes are shared at the class level.
- **Error analysis (seed 0, full pipeline per model):** the confusions are the same for every model:
  Street → InsideCity (47 errors summed over the 5 models), OpenCountry → Mountain (43), InsideCity → Street
  (19), InsideCity → TallBuilding (13), Mountain → Forest (11). 8 images are wrong for all 5 models (7 of
  them OpenCountry). These category boundaries are genuinely ambiguous (A1 found the same confusions with
  full supervision), so no combination of models can fix them.
- **Label-free model selection does not work:** the average confidence of each model on the unlabelled test
  images is inversely related to its accuracy (PE-Core most confident, 0.992, least accurate, 93.25; CLIP
  least confident, 0.984, most accurate, 95.25). So the best single model cannot be identified without
  labels, which is why the final model is the ensemble (D6).


## Step 5 — Class-balance prior in the refinement (DECISIONS.md D8)

**Idea.** The refinement in Step 3 lets a class gather as many test images as look similar to it, so a class
whose starting weights are slightly off can absorb part of a neighbouring class. Adding a uniform class prior
fixes the total "mass" each class may take: in every iteration the soft assignments are rebalanced with three
Sinkhorn steps (`methods.sinkhorn_balance`) so each class receives the same share, while each image still
sums to 1. This assumes the test classes are roughly balanced (here 25 each). The adoption rule (better mean
over the 5 draws of the final ensemble) was written down before running it.

| Linear probe + flip + transductive | CLIP L/14 | DFN-5B | PE-Core | SigLIP | SigLIP 2 | **5-model ensemble** |
|---|---:|---:|---:|---:|---:|---:|
| Unbalanced (Step 3/4) | 95.25 ± 1.2 | 94.7 ± 0.9 | 93.25 ± 0.8 | 94.5 ± 4.2 | 95.0 ± 1.0 | 95.0 ± 1.4 |
| **Balanced** | **96.05 ± 0.6** | 95.65 ± 1.3 | 93.45 ± 0.6 | 95.0 ± 3.8 | 95.8 ± 0.4 | **95.7 ± 1.4** |

Ensemble per draw, balanced: 94.25, 96.5, 94.25, 96.25, 97.25 (unbalanced: 93.75, 96.0, 93.25, 95.75, 96.25).
Zero-shot + refinement (no labelled image), ensemble: 93.5 → 94.0.

- **Adopted** under the pre-set rule: the ensemble mean improves by +0.7, and every model's mean improves
  (+0.2 to +0.9). The final model (draw 0) goes from 93.75% to 94.25%.
- The prior makes the single models more stable (CLIP ± 1.2 → ± 0.6) but **does not rescue SigLIP's bad
  draw 0** (87.3 → 88.2): that failure comes from a wrong starting classifier for some classes, not from
  class sizes.
- The best single model (CLIP, 96.05 ± 0.6) is still slightly above the ensemble (95.7 ± 1.4), as in Step 4;
  the final model remains the ensemble (D6).
