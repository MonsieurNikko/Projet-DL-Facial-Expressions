# Project State — Facial Expression Recognition

Single source of truth for the current state. Update it at the end of each
work cycle. History belongs in `PROJECT_JOURNAL.md`.

## Stable facts

- Students: Nikko (Student A), Simon (Student B). Both must be able to
  explain the whole project orally.
- Deadline: **2026-10-09**.
- Stack: Python, Keras/TensorFlow, Google Colab (GPU) for every run.
- Dataset: FER2013, Kaggle `msambare/fer2013` (approved). License: to verify.
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
- 1a: per-class counts. 1b: example grid + format check.
- 1c: `tf.data` train/val/test, one-hot labels, /255 normalisation,
  prefetch (cache on val/test only, so train is reshuffled every epoch).
- Part 2: dense baseline (Flatten → Dense 256 → Dense 128 → softmax 7,
  623,879 parameters), Adam + categorical cross-entropy, 20 fixed epochs,
  learning curves, validation accuracy vs chance and majority-class references.
- Verified end to end on a synthetic dataset; **still to run in Colab on real data**.

## Progress

- [ ] Part 1 — Dataset research and preparation (code done, Colab run pending)
- [ ] Part 2 — Dense baseline (code done, Colab run pending)
- [ ] Part 3 — CNN
- [ ] Part 4 — Training
- [ ] Part 5 — Evaluation and error analysis
- [ ] Part 6 — At least 3 experiments

Optional, only after Part 6: enrichment, multi-face detection / YOLO, video.

## Current model and results

Dense baseline written; validation accuracy to be measured in Colab.

## Open issues

- Verify the dataset licence on the Kaggle page (required in Part 1).
- Part 1 still lacks a dataset description cell (source, authors, licence) and
  a class-distribution chart; Part 2 lacks the theory cell (weights/biases,
  forward pass, backpropagation, gradient descent).
- Run the notebook in Colab: check the counts printed by 1c and record the
  baseline validation accuracy (and whether the curves show overfitting).

## Next step

Run Parts 1–2 in Colab, then Part 3 — CNN.
