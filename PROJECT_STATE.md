# Project State — Facial Expression Recognition

Single source of truth for the current state. Update it at the end of each
work cycle. History belongs in `PROJECT_JOURNAL.md`.

## Stable facts

- Students: Nikko (Student A), Simon (Student B). Both must be able to
  explain the whole project orally.
- Deadline: **2026-10-09**.
- Stack: Python, Keras/TensorFlow, Google Colab (GPU) for every run.
- Dataset: FER2013, Kaggle `msambare/fer2013` (approved). Licence: Database
  Contents License (DbCL) v1.0 (Kaggle page). Authors: Pierre-Luc Carrier and
  Aaron Courville, ICML 2013 challenge (Goodfellow et al., arXiv:1307.0414).
- Local workspace: `/Users/Nikko/Documents/code/pro`.
- GitHub remote: `origin` = `MonsieurNikko/Projet-DL-Facial-Expressions`
  (**private**), branch `main`. Colab needs GitHub authorization to open it. Ask the humans before pushing.
- Assignment PDF: `docs/` (ignored by Git); summary below.

## Assignment summary (from the PDF)

Course: Fondamentaux du Deep Learning, Master S1 2026-2027 (Hanane Zerdoum).
Defence: Friday 2026-10-09. Deliverables: a commented Colab notebook that
runs end to end, a presentation (approach, architectures, experiments,
results), a demo of the final model. Format/duration of the presentation
not specified.

Grading /20: Part 1 data = 3 · Parts 2–3 baseline + CNN = 5 ·
Parts 4–5 training + evaluation = 3 · Part 6 experiments = 4 ·
Parts 7–9 enrichment = 2 · defence, demo, mastery = 3.

What each part must show or explain:
- Part 1: source (link, authors, licence), total images, classes, images per
  class, image size, format, imbalance, examples per class; resizing,
  normalisation, labels, encoding, train/val/test and the role of each step.
- Part 2: Flatten → Dense + activation → Dense softmax as a reference; explain
  input transformation, weights and biases, forward pass, activations, loss,
  backpropagation, gradient descent, multiclass output.
- Part 3: own CNN, justified; explain filters, convolution, feature maps,
  kernel size, stride, padding, ReLU, pooling, Flatten, Dense, output; a
  layer-by-layer table of shapes and parameter counts.
- Part 4: justify loss, optimiser, batch size, epochs, metrics; train/val
  loss and accuracy curves; identify overfitting.
- Part 5: confusion matrix, per-class performance, most confused classes,
  hardest classes and why; examples "image + true class + predicted class +
  probability", including wrong predictions.
- Part 6: ≥ 3 experiments, few parameters changed at a time; comparison table
  (change, validation result, observation); justify the final model.
- Parts 7–9 (bonus): augmentation, transfer learning, regularisation,
  YOLO face detection (needs a face-specific detector), video.
- Defence: each member must explain the system, the CNN, the main code, the
  choices, results, errors and improvements; questions may target any member.

## Dataset (measured)

- 35,887 images: 28,709 train / 7,178 test. All 48×48, grayscale, `.jpg`.
- Train / test per class:
  angry 3,995 / 958 · disgust 436 / 111 · fear 4,097 / 1,024 ·
  happy 7,215 / 1,774 · neutral 4,965 / 1,233 · sad 4,830 / 1,247 ·
  surprise 3,171 / 831.
- Strong imbalance: `disgust` has ~16× fewer images than `happy`.
- 13 blank images (no face, pixel std < 1): 12 in train (7 in `angry`), 1 in
  test (`test/angry/PublicTest_5543497.jpg`). Removed in step 0 (train and
  test), so the notebook works on 28,697 train / 7,177 test images. Good
  point for the oral (data quality).
- Splits (after removing blank images): train 22,958 / validation 5,739
  (20 % of train, seed 42) / test 7,177 (final evaluation only).

## Notebook — `notebooks/fer2013_expressions.ipynb` (the only notebook)

- Single source of code since 2026-10-08 evening. The former
  `soutenance/notebook_version_detaillee.ipynb` was merged into it and
  deleted; its long explanations are in `soutenance/explications.md` (no
  executable code), with the Part 1 jury questions and answers.
- Step 0: shared constants, global seed, idempotent kagglehub download,
  removal of the 13 blank images (std < 1).
- Part 1: description, 1a counts + bar chart, 1b examples + format check,
  1c `tf.data` train/val/test, one-hot, /255, **batch size 128** (180 updates
  per epoch, last batch 46 images), cache on val/test only.
- Part 2: dense baseline (Flatten, Dense 128 ReLU, Dense 7 softmax), 20 epochs.
- Part 3: `cnn_base` (Conv2D 32 5×5, pool, Conv2D 64 3×3, pool, Flatten,
  Dense 7 softmax, 64,135 parameters), layer table.
- Part 4: choices table, early stopping (max 30, patience 6), curves.
- Part 5: confusion matrix, per-class recall, ROC, top confusions, examples.
- Part 6: `entrainer` (20 fixed epochs, best-epoch weights, optional
  ReduceLROnPlateau) and `mesurer`. Chain: E0 LR schedule, E1 Dense 128 + 64,
  E1 bis BatchNorm, E2 Dropout 0.3, E3 4 conv blocks, E4 augmentation (flip,
  rotation, zoom, Gaussian blur), **E5 = E4 + class_weight** (compared with
  E4). E0 and E1 bis gave no gain alone but stay in the later models (choice
  made before the results, stated in the readings). 6g prints the table plus
  per-class precision and F1. **6h picks the final model automatically: best
  validation accuracy** (rule announced at the start of Part 6). 6i: single
  test evaluation, model saved; demo cell.
- **Run on real data on 2026-10-08 evening (Google Colab, GPU T4), no error;
  the notebook in the repo contains these outputs** (45 outputs). Final model
  chosen by 6h: `e4_augmentation`.

## Progress

- [x] Part 1 — Dataset research and preparation
- [x] Part 2 — Dense baseline
- [x] Part 3 — CNN
- [x] Part 4 — Training (choices justified, curves, overfitting comment)
- [x] Part 5 — Evaluation and error analysis
- [x] Part 6 — experiments E0–E5, table, final model, test evaluation, demo

Optional, only after Part 6: enrichment, multi-face detection / YOLO, video.
Presentation slides: not started.

## Current model and results

Reference run: 2026-10-08 evening, Google Colab, GPU T4, seed 42, batch 128,
blank images removed (train 22,958 / val 5,739 / test 7,177). Outputs saved
in the notebook.

| Model | Change | Val acc | Val loss | Best epoch | Train−val gap | `disgust` recall | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Dense baseline | Part 2 | 34.2 % | | 20 fixed | | | reference |
| `cnn_base` | Part 3 (early stopping) | 50.9 % | 1.335 | 10 / 16 | +7.6 | 18.6 % | reference |
| E0 | + ReduceLROnPlateau | 50.5 % | 1.327 | 8 / 20 | +6.5 | 18.6 % | no gain, kept in chain |
| E1 | + Dense 128 + 64 | 50.7 % | 1.326 | 6 / 20 | +7.3 | 24.3 % | kept |
| E1 bis | + BatchNorm | 52.1 % | 1.349 | 11 / 20 | +18.9 | 30.0 % | kept (rule) |
| E2 | + Dropout 0.3 | 53.5 % | 1.281 | 14 / 20 | +12.8 | 34.3 % | kept |
| E3 | + conv blocks 128 and 256 | 51.1 % | 1.303 | 4 / 20 | +5.5 | 0.0 % | should be rejected; E4 built on it anyway |
| E4 | + augmentation | 60.2 % | 1.065 | 19 / 20 | +3.9 | 20.0 % | **final model** |
| E5 | E4 + class_weight | 54.7 % | 1.188 | 17 / 20 | +0.7 | 48.6 % | rejected (best disgust F1) |

- References: chance 14.3 %, always `happy` 24.4 % (val), 24.7 % (test).
- `disgust` precision / F1: E4 73.7 / 31.5 %, E5 27.9 / 35.4 %.
- E3 detail: val accuracy keeps rising to ~58 % at epoch 20 while val_loss
  rises (overconfidence); best weights are restored on val_loss (epoch 4).
- `cnn_base` val recall: angry 43.0 · disgust 18.6 · fear 24.8 · happy 74.8 ·
  neutral 50.3 · sad 38.9 · surprise 63.5 %. Top confusions: sad→neutral 187,
  neutral→sad 174, fear→sad 166, sad→happy 150, neutral→happy 142.
- Test (used once, E4): **61.2 %** (loss 1.049). Recall: angry 53.7 ·
  disgust 20.7 (23/111) · fear 33.2 · happy 80.6 · neutral 74.0 · sad 46.7 ·
  surprise 71.1 %.
- Demo cell: a `happy` test image predicted `happy` at 0.98.

## Branches (2026-10-07)

- `main`: contains Parts 1–6. Branch `simon` (commit `74b1445`, Parts
  3–5) was reviewed and fast-forwarded into local `main`; the completed
  notebook and documentation update was pushed to `origin/main` after approval.
- `refactor/part1-agent-rules`: already contained in `main`, nothing to merge.
- `claude/add-claude-skills`: older version superseded by `main`; merging it
  would remove content (dataset description, student names). Do not merge;
  can be deleted.

## Open issues

- Be ready to explain at the oral: E0 and E3 did not improve accuracy but stay
  in the chain (written before the run); E1 bis kept by the rule despite a
  worse loss; best epoch chosen on val_loss but final model on val accuracy
  (E3 shows the two can disagree).
- E4 and E5 were still improving at epoch 20: longer training not tested.
  Milder class weights (square root) not tested.
- Results come from one seed: differences under 1 point are not meaningful.
- Presentation slides not started.
## Next step

Prepare the presentation and rehearse the demo; mock jury on Parts 2–6
(`soutenance/explications.md`).
