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
- Part 6, section "6f bis" (added 2026-10-08, **written but never run**):
  four more experiments taken from the CNN of Simon's earlier PyTorch
  project (`soutenance/Project_MTH416.ipynb` + `MTH416_Report.pdf`), ported
  to Keras, each built on the previous one starting from E3:
  E6 BatchNormalization after each convolution (356,743 parameters) ·
  E7 4th block Conv2D(256) with `padding="same"` (685,703 parameters,
  function `creer_cnn_4_blocs`) · E8 same model, `ReduceLROnPlateau`
  (factor 0.5, patience 2) + early-stopping patience 6, max 50 epochs
  (function `entrainer_long`) · E9 E8 + augmentation (flip, rotation 0.03,
  translation 0.05). 6h now picks the final model in code: best validation
  accuracy among E3, E6–E9. Parameter counts and shapes were checked by
  building the models only.
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
  (local run; Colab check pending). E6–E9 added, not run yet.

Optional, only after Part 6: enrichment, multi-face detection / YOLO, video.
Presentation slides: first version done (2026-10-08), 17 slides, online deck
(private, owner Simon): https://claude.ai/artifact/Msi5ypa6RFzgdRpXPDbLLm
Short version chosen for the defence (8 slides, one per part, Gamma, Simon's
account): https://gamma.app/docs/ourc43y9pgqii3z
Numbers come from the table below. Two image placeholders to fill from the
Colab run: CNN curves (4b) and example grid (5d). Speaker notes on each slide.

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

- Detailed notebook only, 2026-10-08: experiment E3b added after E3
  (**written, never run**): same model as the modified E3, trained with
  `ReduceLROnPlateau` (factor 0.5, patience 1) through a new `baisse_lr`
  argument of `entrainer`; early stopping unchanged (patience 3).
- Detailed notebook only (`soutenance/`), 2026-10-08: E3 was changed in
  place and no longer matches the main notebook or the dated tables. First
  convolution is 5×5 (Simon's edit) and the third block has a second
  `Conv2D(128, 3, padding="same")` (503,943 parameters, not run). E4, E5 and
  E5b there still copy the old E3 architecture (3×3 first, single conv).
- E5b (added 2026-10-08, **written, never run**, in both notebooks, right
  after E5): same model as E3 trained on a training set rebalanced by random
  oversampling (every class copied up to the size of `happy`, about 40,000
  images per epoch; validation untouched). `entrainer` got a `donnees_train`
  argument for it, plus an optional number of epochs (`epoques` in the main
  notebook, `epoch` in the detailed one; fixed epochs = no early stopping).
  Undersampling and SMOTE were considered and not implemented (reasons in the
  cell). In the main notebook E5b is a candidate for the final model in 6h.
- E6–E9 have no results: the students must run them, write the reading cell
  at the end of "6f bis", and add their lines to the dated table in 6g.
  If the final model is no longer E3, the dated test numbers in 6i, the
  README results and the slides must be updated too.
- E8 and E9 train much longer (up to 50 epochs on a model about 2.5× heavier
  than E3): expect a clearly longer notebook run on CPU.
- Partial trial on Simon's Mac (Apple M4, CPU, TF 2.21, seed 42, same
  protocol, 2026-10-08, outside the notebook): E3 56.8 % / loss 1.130;
  E3 + BatchNormalization 57.6 % / loss 1.145, with a jumpy validation
  curve. So E6 alone is not a clear gain there. E7–E9 not measured.
- Not ported from the MTH416 model: SiLU, 5×5 first kernel, stepped dense
  head, SGD with momentum, z-score normalisation.
- `soutenance/notebook_version_detaillee.ipynb` does not contain E6–E9.
- Rule for assistants on Simon's side: do not train models; write the code,
  the students run it.
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

Students run the notebook in Colab (including E6–E9), complete the
"6f bis" reading and the 6g table, and review Parts 4–6; then prepare the
presentation (approach, architectures, experiment table, results, errors)
and rehearse the demo cell. Commit after review (Conventional Commits),
push after approval.
