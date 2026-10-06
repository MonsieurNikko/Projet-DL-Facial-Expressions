# Project State — Facial Expression Recognition

Single source of truth for the current state. Update it at the end of each
work cycle. History belongs in `PROJECT_JOURNAL.md`.

## Stable facts

- Students: Nikko (Student A), Simon (Student B). Both must be able to
  explain the whole project orally.
- Deadline: **2026-10-09**.
- Stack: Python, Keras/TensorFlow, Google Colab (GPU) for every run.
- Dataset: FER2013, Kaggle `msambare/fer2013` (approved). License: to verify.
- Deliverables: commented Colab notebook, presentation, final model demo.
- Local workspace: `/Users/Nikko/Documents/code/pro`.
- GitHub remote: `origin` = `MonsieurNikko/Projet-DL-Facial-Expressions`,
  branch `main`. Ask the humans before pushing.
- Assignment PDF: `docs/` (ignored by Git).

## Dataset (measured)

- 35,887 images: 28,709 train / 7,178 test. All 48×48, grayscale, `.jpg`.
- Train / test per class:
  angry 3,995 / 958 · disgust 436 / 111 · fear 4,097 / 1,024 ·
  happy 7,215 / 1,774 · neutral 4,965 / 1,233 · sad 4,830 / 1,247 ·
  surprise 3,171 / 831.
- Strong imbalance: `disgust` has ~16× fewer images than `happy`.
- Splits: train 22,968 / validation 5,741 (20 % of train, seed 42) /
  test 7,178 (final evaluation only).

## Notebook — `notebooks/fer2013_expressions.ipynb`

- Step 0: shared constants, global seed, idempotent kagglehub download
  (works locally and on Colab without a clone).
- 1a: per-class counts. 1b: example grid + format check.
- 1c: `tf.data` train/val/test, one-hot labels, /255 normalisation,
  cache + prefetch.
- Verified on a synthetic dataset; **still to run in Colab on real data**.

## Progress

- [ ] Part 1 — Dataset research and preparation (code done, Colab run pending)
- [ ] Part 2 — Dense baseline
- [ ] Part 3 — CNN
- [ ] Part 4 — Training
- [ ] Part 5 — Evaluation and error analysis
- [ ] Part 6 — At least 3 experiments

Optional, only after Part 6: enrichment, multi-face detection / YOLO, video.

## Current model and results

None yet.

## Open issues

- Verify the dataset license.
- Run the notebook once in Colab and check the counts printed by 1c.

## Next step

Run Part 1 in Colab, then Part 2 — dense baseline.
