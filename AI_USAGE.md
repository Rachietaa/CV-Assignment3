# AI Usage

## 1. AI coding tool used

**Claude Code** (Anthropic), used inside VS Code as a pair programmer for the whole project.

The AI wrote code I asked for, explained one-shot and test-time adaptation methods, and **suggested** options.
I decided the rules for data use, how the single shot per class is chosen, which methods to try and how to
compare them, interpreted the results, and made every Git commit myself. Each decision and its reason is in
[DECISIONS.md](DECISIONS.md); every result is in [EXPERIMENTS.md](EXPERIMENTS.md).

---

## 2. Representative examples of how AI assisted me

**1. Making the one-shot rule visible in the code.** At my request the AI wrote the data layer so that only
the 16 listed shot images and the test images are ever opened ([src/data.py](src/data.py)); shot lists are
chosen at random with fixed seeds and saved as files, so anyone can check that exactly one image per class
is used.

**2. Implementing the methods.** Wrappers for five foundation models (open_clip), prompt sets, image
prototypes, text-image fusion, Tip-Adapter, the text-initialized linear probe, transductive refinement and
its class-balanced version, and the joint-space ensemble of the five models.

**3. Analysis.** A cross-model error analysis (which test images every model gets wrong, and which classes
are confused), and a check of whether a model can be chosen without labels from its confidence on the
unlabelled test images. Both shaped my final decisions (D6).

**4. Reproducibility.** A final classifier file and an inference script that re-encodes raw images and
reproduces the reported accuracy exactly (94.25%, 377/400).

---

## 3. Incorrect, ineffective or questionable AI suggestions

**Incorrect: OpenAI CLIP loaded with the wrong activation.** The AI's first model list loaded OpenAI's CLIP
weights into a configuration with standard GELU, while those weights were trained with QuickGELU. A library
warning revealed it. With the fix, CLIP's zero-shot accuracy went from 89.25% to 91.25%, and it became the
best zero-shot model.

**Incorrect: a garbled line of code.** The first version of the Tip-Adapter function had a nonsensical
return statement (terms multiplied by zero). It was caught when the AI re-read the function before running
it, and rewritten.

**Ineffective: methods that did not beat zero-shot.** Tip-Adapter with its paper defaults moved accuracy by
only ±0.2 over zero-shot, and a 50/50 mix of text and image prototypes was *worse* than zero-shot for most
models. The five-model ensemble the AI planned also did not beat the best single model.

**Questionable: progress output and time estimates.** The extraction script buffered its output, so my
terminal showed only a warning for 20 minutes and I thought it was stuck; the AI's time estimates for the
model downloads were off by about 30 minutes.

---

## 4. How I verified or modified the AI-generated solutions

| Issue | How it was verified | Change made |
|---|---|---|
| Wrong CLIP activation | Library warning; re-extracted CLIP and compared | `force_quick_gelu` for OpenAI weights; all CLIP numbers re-run |
| Every new pipeline step | CPU smoke test with a small CLIP model before the long runs | Bugs fixed before any reported number |
| Any test-set selection | No labelled validation set exists, so settings were paper defaults or fixed in advance; the D8 adoption rule was written down before running it | Final model is the ensemble (D6) and draw 0 (D7), not the best-scoring choices |
| Shipped model = reported number | `predict.py` on the raw test images | Reproduces 94.25% (377/400) exactly |
| Numbers in the write-up | Recomputed from the saved results files | Corrected two over-general claims (gain on every draw; a mixing-curve peak measured before the CLIP fix) |

---

## 5. Decisions I made rather than accepting the AI's recommendation

### Trying one more improvement after the final model was chosen

After the five-model ensemble was fixed as the final model, the AI listed a class-balance prior for the
refinement as an optional experiment and recommended stopping. I asked whether we could improve further and
chose to run it, with the adoption rule fixed beforehand. It improved every model and raised the final
pipeline from 95.0 ± 1.4 to 95.7 ± 1.4.

### Not chasing an unrealistic target by bending the rules

I asked for at least 98%. Once the AI showed that 8 test images are wrong for all five models and that even
Assignment 1's fully supervised model reached 95.75%, I accepted that 98% would only be reachable by tuning on
the test set or using more labelled images, and kept the honest protocol.
