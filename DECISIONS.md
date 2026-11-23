# Decision Log

Every design decision in this assignment, why it was made, and what evidence supported it.
This feeds the report sections "Foundation model / training recipe" and "Conclusion".

## Fixed by the assignment rules

| Rule | Consequence |
|---|---|
| Only one labelled training image per class | 16 labelled images in total; no other training image is read by the code, labelled or not. |
| Foundation models are allowed (and encouraged) | Unlike Assignment 2, pretrained CLIP / SigLIP weights may be used. |
| Test-time adaptation is suggested | Unlabelled test images may be used at prediction time, never their labels. |

## Carried over from Assignments 1 and 2

| Decision | Why (A1/A2 evidence) |
|---|---|
| Test set used for reporting, not for choosing methods or settings | Keeps the test accuracy an honest estimate (A1, A2). |
| Repeat over several random draws / seeds before trusting a difference | In A2 a single seed overstated the SSL gain by 60% (+4.6 vs +2.9). |
| Every claimed gain needs a matched control | In A2 a regularized supervised control removed an apparent +4.4 SSL gain. |

## Decisions made in Assignment 3

| # | Decision | Options considered | Choice | Why | Evidence / result |
|---|---|---|---|---|---|
| D1 | Data used besides the 16 shots | (a) unlabelled test images at test time only; (b) also the other 2,384 training images, unlabelled; (c) nothing else | **(a)** | The PDF suggests test-time adaptation, which only needs the unlabelled test images. Reading any other training image, even without its label, risks violating "strictly forbidden to use more than one training sample per class". | |
| D2 | Choice of the one image per class | (a) random with a fixed seed, repeated over 5 draws; (b) hand-picked "typical" images | **(a)** | Hand-picking means looking through other training images, which already uses more than one sample per class. Random draws are fair, reproducible, and show how much the choice of the single image matters. Draw 0 is the main draw. | |
| D3 | Setting hyperparameters without a validation set | (a) paper defaults, tuning only on the 16 shots (leave-one-out), sensitivity reported; (b) tuning on test | **(a)** | There is no legal labelled validation data: other training labels are forbidden and tuning on test would inflate the result. | |
| D4 | Colour | (a) images as RGB, as stored; (b) everything converted to grayscale | **(a)** | Foundation models are trained on colour images. In A1/A2 colour was a training *shortcut* (only Flower is in colour); here it is real, available information in the test images. | |
| D5 | Setup | New repo + environment | **Repo `CV-Assignment3`, its own `.venv`** (needs `open_clip`, which A1/A2's environment does not have); compute on the Mac (Apple M5) | Frozen models need only forward passes over ~500 images. | |
| D6 | Final model | (a) all 5 foundation models in one joint-space ensemble; (b) the best single model on test (SigLIP 2 / CLIP); (c) a model chosen without labels by its confidence on the test images | **(a): ensemble of CLIP L/14, DFN-5B H/14, PE-Core L/14, SigLIP and SigLIP 2, each with the text-initialized linear probe; flip test-time augmentation and transductive refinement in the joint space** | There is no labelled validation set, so choosing the best single model would be choosing on test. The plan (written before any results) was to combine all models. Label-free selection was checked and does not work: the most confident model (PE-Core) is the least accurate. Cost: 95.0 ± 1.4 vs 95.35 for the best single model. | Ensemble 95.0 ± 1.4 (5 draws); single models 93.25–95.35 |
