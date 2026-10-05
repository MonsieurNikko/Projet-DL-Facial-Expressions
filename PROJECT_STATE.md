# Project State — Facial Expression Recognition

## Goal

Build and understand a Deep Learning system for facial-expression recognition
for the university project.

Final deliverables:
- commented Google Colab notebook;
- presentation;
- final model demonstration.

Both students must understand and be able to explain the project.


## Current phase

Phase 1 — Dataset research and understanding.


## Current microtask

Part 1 / micro-step 1c — build train/val/test `tf.data` datasets (grayscale, one-hot, 80/20 split seed 42, /255 normalisation). Written; awaiting execution in Colab (TensorFlow not installed locally). Expected: batch (64, 48, 48, 1) / (64, 7), pixels in [0, 1], 22,968 / 5,741 / 7,178 images.
Done: 1a (inventory) and 1b (example grid + format check, Copilot PASS after unreadable-file fix).
Colab run owner: Student B (Simon); reviewer: Student A (Nikko).


## Assignment progress

- [ ] Part 1 — Dataset research and preparation
- [ ] Part 2 — Dense baseline
- [ ] Part 3 — CNN
- [ ] Part 4 — Training
- [ ] Part 5 — Evaluation and error analysis
- [ ] Part 6 — At least 3 experiments

Optional later:
- [ ] Part 7 — Enrichment
- [ ] Part 8 — Multi-face detection / YOLO
- [ ] Part 9 — Video


## Dataset

Status: APPROVED — FER2013 (Facial Expression Recognition using FER2013)

Source:
`msambare/fer2013`

License:
TBD

Number of images:
35,887 total — 28,709 train and 7,178 test (verified by executing the notebook cell).

Classes and image counts (train / test):
- angry: 3,995 / 958
- disgust: 436 / 111
- fear: 4,097 / 1,024
- happy: 7,215 / 1,774
- neutral: 4,965 / 1,233
- sad: 4,830 / 1,247
- surprise: 3,171 / 831

Image dimensions:
All 35,887 images are 48×48, grayscale (PIL mode L), `.jpg` (verified in 1b). Network input: (48, 48, 1).

Class balance:
Imbalanced; `disgust` is the smallest class and `happy` the largest in both splits. No model impact has been measured yet.


## Current model

None yet.


## Current results

No training performed yet.


## Experiments

No experiments yet.


## Important decisions

- FER2013 (`msambare/fer2013`) approved by the students.
- The notebook is built progressively, one cell per microtask. The premature skeleton was removed (Copilot PASS).
- Secrets and data are ignored by Git.


## Repository

- Public GitHub remote `origin` = `MonsieurNikko/Projet-DL-Facial-Expressions`, branch `main` (commits `feeb05b` 1a, `4e9bb38` 1b).
- `notebooks/fer2013_expressions.ipynb` holds steps 1a–1c; Simon works in `notebooks/part3_cnn.ipynb`.
- `data/fer2013/` is extracted locally and ignored by Git; the train/test image counts are recorded above.
- `data/raw/fer2013.zip` is the only ZIP archive currently present in `data/raw/`; the three other archives were removed.
- The assignment PDF is in `docs/` and ignored by Git (`docs/*.pdf`).
- `.gitignore` covers `.env`, `kaggle.json`, `data/`, `models/`, `*.zip`, caches, and `.DS_Store`.


## Student responsibilities

### Student A
Current responsibility:
TBD

Concepts already understood:
TBD


### Student B
Current responsibility:
TBD

Concepts already understood:
TBD


## Open issues

- License details remain to be verified in Part 1.
- The reference commit `95fcc08` provides a Git baseline for reviews.
- The Colab download method (Kaggle API or Drive upload) has not been decided; it is not needed for local micro-step 1a.



## Next candidate step

Candidate: validate 1c in Colab, commit 1b+1c, then Part 2 — dense baseline. Wait for the next user `go` before implementation.