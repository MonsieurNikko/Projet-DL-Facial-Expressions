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

## Notebook — `notebooks/fer2013_expressions.ipynb`

- Step 0: shared constants, global seed, idempotent kagglehub download
  (works locally and on Colab without a clone).
- Part 1 description cell: source, authors, licence, collection, counts,
  classes, size, format, imbalance.
- 1a: per-class counts + class-distribution bar chart. 1b: example grid + format check.
- 1c: `tf.data` train/val/test, one-hot labels, /255 normalisation,
  prefetch (cache on val/test only, so train is reshuffled every epoch).
- Part 2: dense baseline as in the assignment (Flatten → Dense 128 ReLU →
  Dense 7 softmax, 295,943 parameters), Adam + categorical cross-entropy,
  20 fixed epochs; theory cell (weights/biases, forward pass, activations,
  loss, backpropagation, gradient descent, multiclass output); learning
  curves; validation accuracy vs chance and majority class; 2d shows the 7
  probabilities for one validation image.
- Part 3: CNN as in the assignment diagram (Conv2D 32 3×3 ReLU →
  MaxPooling 2 → Conv2D 64 3×3 ReLU → MaxPooling 2 → Flatten → Dense 7
  softmax, 63,623 parameters confirmed by `summary()`), same compilation as
  the baseline; theory cell (filter, kernel size, convolution, feature map,
  stride, padding, ReLU, pooling, Flatten, Dense, output), layer-by-layer
  table and justification. 3a resets the seed just before building the
  model, like every Part 6 experiment, so all CNNs start from the same state.
- Part 4: table justifying loss, optimiser (Adam, lr 0.001), batch size 64,
  epochs (max 30 + early stopping) and metrics (accuracy, then per-class
  recall); 4a CNN training with early stopping on `val_loss` (patience 3,
  best weights restored); 4b learning curves (function `tracer_courbes`,
  reused in Part 6) + overfitting interpretation.
- Part 5: 5a confusion matrix and per-class recall; 5b one-vs-rest ROC/AUC;
  5c top-5 confusions and classes sorted by recall + why; 5d grid of 4 correct
  and the 8 most confident wrong predictions (true class, predicted class,
  probabilities) + error analysis. All on validation.
- Part 6: helpers `entrainer` / `mesurer`; 5 experiments, each changing one
  thing from the last retained model (E1 Dense 128, E2 Dropout 0.5, E3 third
  conv block, E4 augmentation, E5 class weights); comparison table printed
  and in Markdown; final choice E3; its curves; single test evaluation;
  model saved to `models/modele_final.keras`; demo cell (load model, one
  image, 7 probabilities).
- Whole notebook executed end to end on real data, locally on CPU (Windows,
  TensorFlow 2.21, about 9 minutes, 543 s); same numbers on two consecutive
  runs. **Not yet run in Colab.** Saved kernel is now the neutral `python3`.

## Progress

- [ ] Part 1 — Dataset research and preparation (code done, Colab run pending)
- [ ] Part 2 — Dense baseline (code done, Colab run pending)
- [x] Part 3 — CNN
- [x] Part 4 — Training (choices justified, curves, overfitting comment)
- [x] Part 5 — Evaluation and error analysis
- [x] Part 6 — 5 experiments, table, final model, test evaluation, demo
  (local run; Colab check pending)

Optional, only after Part 6: enrichment, multi-face detection / YOLO, video.
Presentation slides: not started.

## Current model and results

Single local run (seed 42, CPU, Windows, TF 2.21), 2026-10-08, after
removing the 13 blank images in step 0. To re-check on Colab (GPU may give
slightly different numbers).

Validation (5,739 images; train 22,958):

| Model | Change | Val acc | Val loss | Best epoch | Train−val gap | `disgust` recall | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Dense baseline | Part 2 | 34.0 % | | 20 fixed | | | reference |
| `cnn_base` | Part 3 | 50.8 % | 1.339 | 8 / 11 | +6.2 | 11.4 % | reference |
| E1 | + Dense(128) | 51.9 % | 1.303 | 6 / 9 | +7.9 | 12.9 % | kept (+1.1) |
| E2 | + Dropout(0.5) before Dense | 52.2 % | 1.274 | 7 / 10 | +3.6 | 18.6 % | kept (tie, lower loss and gap) |
| E3 | + 3rd block Conv2D(128) | 58.3 % | 1.127 | 14 / 17 | +3.4 | 25.7 % | final model |
| E4 | E3 + augmentation | 56.6 % | 1.144 | 20 / 23 | −3.0 | 10.0 % | rejected |
| E5 | E3 + class_weight | 53.8 % | 1.238 | 15 / 18 | +1.7 | 60.0 % | rejected |

- References: chance 14.3 %, always `happy` 24.4 % (validation).
- `cnn_base` recall: angry 35.9 · disgust 11.4 (8 / 70) · fear 28.2 ·
  happy 72.7 · neutral 48.8 · sad 45.5 · surprise 64.4 %. Top confusions:
  neutral→sad 203, fear→sad 183, sad→neutral 167, angry→sad 152,
  neutral→happy 132.
- E3 recall: angry 47.0 · disgust 25.7 · fear 29.4 · happy 82.9 · neutral
  59.8 · sad 47.3 · surprise 72.0 %. E5 raises disgust to 60.0 % but lowers
  happy (69.1), neutral (55.3), sad (44.9) and angry (45.0).

Test set (used once, final model E3 `e3_3_blocs`, 355,847 parameters):
57.7 % accuracy (loss 1.124, 7,177 images; always `happy` = 24.7 %).
Recall: angry 50.1 · disgust 35.1 (39 / 111) · fear 27.5 · happy 81.8 ·
neutral 57.9 · sad 46.0 · surprise 72.4 %.

## Branches (2026-10-07)

- `main`: contains Parts 1–6. Branch `simon` (commit `74b1445`, Parts
  3–5) was reviewed and fast-forwarded into local `main`; the completed
  notebook and documentation update was pushed to `origin/main` after approval.
- `refactor/part1-agent-rules`: already contained in `main`, nothing to merge.
- `claude/add-claude-skills`: older version superseded by `main`; merging it
  would remove content (dataset description, student names). Do not merge;
  can be deleted.

## Open issues

- Changes of 2026-10-08 (Simon), in both notebooks, **written but never
  run**: (1) dropout rate 0.5 → 0.3 in E2–E5, with the justification in the
  E2 cell; (2) `RandomGaussianBlur(factor=0.5, kernel_size=3, sigma=1.0,
  value_range=(0, 1))` added to the E4 augmentation (needs a recent Keras 3;
  present in 3.15, to check on Colab); (3) new experiment E0 before E1:
  `cnn_base` trained with `ReduceLROnPlateau` (factor 0.5, patience 1) through
  a new `baisse_lr` argument of `entrainer`; E1–E5 now all pass
  `baisse_lr=True` (Simon's request, before any E0 result).
  (4) early-stopping patience 3 → 6 in 4a, shared by `cnn_base` and every
  Part 6 experiment through `entrainer`.
  (5) architecture: a `Dense(64)` after `Dense(128)` in E1–E5 (E1/E2:
  846,855 parameters); a 4th block `Conv2D(256, 3, padding="same")` +
  `MaxPooling2D(2)` in E3–E5 (527,751 parameters, Flatten 1,024; model
  renamed `e3_4_blocs`). E1 and E3 therefore each add two layers at once.
  (6) new experiment E6 after E5 ("6f bis"): E3 with a `BatchNormalization`
  after each convolution (conv → BN → ReLU → pooling), 529,671 parameters,
  `baisse_lr=True`. 6h still fixes `modele_e3` as the final model; switch it
  to E6 by hand if E6 wins. Justifications added in Markdown for patience 6
  (Part 4 choices) and for using the scheduler in E1–E6 (Part 6 intro).
  (7) Markdown pass: every reading cell of Part 6 (E1–E5, 6h, 6i) now opens
  with a line saying it describes the earlier reference run and must be
  re-read after a new run; E0 and E6 readings are placeholders.
  (8) scheduler made less aggressive after Simon reported runs stopping too
  early: `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)` instead of
  patience 1 with no floor. Diagnosis from the design and an earlier log, not
  from Simon's latest run (its outputs were not saved).
  (9) `entrainer(..., epoques=20)`: every Part 6 experiment now trains a
  fixed 20 epochs and keeps the best-epoch weights (an `EarlyStopping` with
  patience = epochs never fires and only restores the best weights; relies on
  Keras 3 behaviour). `epoques=None` gives back the Part 4 early stopping.
  `cnn_base` still uses early stopping (max 30, patience 6).
  (10) batch normalisation moved: the E6 step of item (6) is removed and
  replaced by "E1 bis" right after E1 (E1 + BN after each convolution,
  847,239 parameters). E2–E5 are built on it and all contain BN (E2:
  847,239; E3–E5: 529,671). Final model in 6h is still `modele_e3`.
  (11) first convolution of every CNN (Part 3 `cnn_base` and E0–E5) changed
  from 3×3 to 5×5. Shapes become 48 → 44 → 22 → 20 → 10 (Flatten unchanged).
  Parameter counts now: `cnn_base` / E0 64,135 · E1 847,367 · E1 bis / E2
  847,751 · E3–E5 530,183. These supersede the counts quoted in items above.
  Part 3 text, layer table and the 5×5 justification updated in both notebooks.
  (12) 6g now also prints per-class precision and F1 for every model
  (`precision_score` / `f1_score`, `zero_division=0`), from a `predictions`
  dict filled by `mesurer`. Tested on fake predictions only.
  Consequences: every measured number below, the dated tables in 6g / 6i, the
  README results and the slides were obtained with dropout 0.5, without blur
  and without E0, and must be redone after a full run. The reading cells of
  E2–E5 describe the old run; the E0 reading cell is a placeholder.
- Run the notebook in Colab (Runtime → Run all; about 9 min locally on CPU,
  probably faster on GPU). The notebook text no longer quotes run numbers, so
  it stays valid if they move; only check that the trends hold (E3 best,
  E4 below E3, E5 trades accuracy for `disgust` recall).
- Parts 4–6 written without a student run: Nikko and Simon must read and be
  able to explain every new cell (experiment chain, final choice, test cell).
- Results come from one seed: differences under 1 point (cnn_base vs E1)
  are not meaningful.
- E4 (augmentation) was stopped by patience 3 while still learning slowly;
  a longer run (more epochs / patience) was not tested. Milder class weights
  (e.g. square root) for `disgust` not tested either.
- Commit `74b1445` ("PART 3-4-5") does not follow Conventional Commits;
  keep the format for the next commits.

## Next step

Students run the notebook in Colab and review Parts 4–6; then prepare the
presentation (approach, architectures, experiment table, results, errors)
and rehearse the demo cell. Commit after review (Conventional Commits),
push after approval.
