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
- Splits: train 22,968 / validation 5,741 (20 % of train, seed 42) /
  test 7,178 (final evaluation only).

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

Single local run (seed 42, CPU, Windows, TF 2.21), 2026-10-07. To re-check
on Colab (GPU may give slightly different numbers).

Validation (5,741 images):

| Model | Change | Val acc | Val loss | Best epoch | Train−val gap | `disgust` recall | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Dense baseline | Part 2 | 36.1 % | | 20 fixed | | | reference |
| `cnn_base` | Part 3 | 50.3 % | 1.322 | 7 / 10 | +5.8 | 13.7 % | reference |
| E1 | + Dense(128) | 49.8 % | 1.309 | 5 / 8 | +6.0 | 11.0 % | kept (tie, lower loss) |
| E2 | + Dropout(0.5) before Dense | 52.3 % | 1.235 | 9 / 12 | +6.6 | 17.8 % | kept |
| E3 | + 3rd block Conv2D(128) | **57.7 %** | **1.126** | 18 / 21 | +7.8 | 34.2 % | **final model** |
| E4 | E3 + augmentation | 55.4 % | 1.161 | 16 / 19 | −1.7 | 6.8 % | rejected |
| E5 | E3 + class_weight | 53.9 % | 1.218 | 20 / 23 | +7.6 | 57.5 % | rejected |

- References: chance 14.3 %, always `happy` 24.4 % (validation).
- `cnn_base` recall: angry 36.6 · disgust 13.7 · fear 25.0 · happy 73.5 ·
  neutral 50.4 · sad 44.4 · surprise 60.5 %. Top confusions: neutral→sad 214,
  fear→sad 192, sad→neutral 182, angry→sad 167, sad→happy 152.
- E3 recall: angry 46.6 · disgust 34.2 · fear 32.6 · happy 82.0 · neutral
  57.1 · sad 49.0 · surprise 66.4 %. E5 raises disgust to 57.5 % but lowers
  happy (73.4), angry (41.4) and fear (28.0).

Test set (used once, final model E3 `e3_3_blocs`, 355,847 parameters):
**59.3 %** accuracy (loss 1.112, 7,178 images; always `happy` = 24.7 %).
Recall: angry 48.7 · disgust 33.3 (37 / 111) · fear 35.9 · happy 84.0 ·
neutral 58.0 · sad 47.8 · surprise 70.5 %.

## Branches (2026-10-07)

- `main`: contains Parts 1–6. Branch `simon` (commit `74b1445`, Parts
  3–5) was reviewed and fast-forwarded into local `main`; the completed
  notebook and documentation update was pushed to `origin/main` after approval.
- `refactor/part1-agent-rules`: already contained in `main`, nothing to merge.
- `claude/add-claude-skills`: older version superseded by `main`; merging it
  would remove content (dataset description, student names). Do not merge;
  can be deleted.

## Open issues

- Run the notebook in Colab (Runtime → Run all; about 9 min locally on CPU,
  probably faster on GPU) and check the numbers above; update the notebook
  text if they move.
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
