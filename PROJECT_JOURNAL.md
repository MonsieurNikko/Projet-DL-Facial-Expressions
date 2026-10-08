# Project Journal — Facial Expression Recognition

Purpose:
Maintain a concise chronological history of validated project work.

This file is NOT a raw chat log.

Only record meaningful validated milestones, decisions, experiments,
results and corrections.

It will later be used to:
- reconstruct the project journey;
- prepare the oral presentation;
- explain why decisions were made;
- show experiments and improvements;
- divide presentation sections between both students.

---

## Entry format

### YYYY-MM-DD — <short title>

**Phase:**  
Part X — ...

**Goal:**  
What were we trying to achieve?

**Owner:**  
Student A / Student B / Both

**What we did:**  
Short factual summary.

**Why:**  
Reason for the chosen approach.

**Important code/concept:**  
Function, notebook cell, model block or Deep Learning concept involved.

**Result:**  
Measured result or concrete outcome.

**Decision:**  
KEEP / REJECT / CONTINUE / INVESTIGATE

**Problems encountered:**  
Only important problems and how they were solved.

**Reviewer:**  
Copilot: PASS / NEEDS_DISCUSSION

**Presentation material:**  
1–3 points potentially useful for the final slides or oral defence.

---
---
### 2026-10-05 — Dataset Selection

**Phase:**  
Part 1 — Dataset research and understanding.

**Goal:**  
Select a public facial expression recognition dataset for the project.

**Owner:**  
Both

**What we did:**  
Selected FER2013 dataset as the primary data source for facial expression recognition.

**Why:**  
FER2013 is a well-established, publicly available dataset. **NOTE: La source initiale (Kaggle abellamit) est incorrecte ; source correcte est `msambare/fer2013`.**

**Important code/concept:**  
None yet - dataset selection is a research step.

**Result:**  
FER2013 dataset selected ; **à mesurer en Partie 1** (source : `msambare/fer2013`).

**Decision:**  
KEEP (source corrigée en msambare/fer2013)

**Problems encountered:**  
- Initial source link incorrect ; corrected to `msambare/fer2013`.
- Chiffres initiaux erronés (3 expressions, 3 582 images, ratio 4:1) ; **à mesurer en Partie 1**.

**Reviewer:**  
En attente de mesures réelles en Partie 1.

**Presentation material:**  
**à mesurer en Partie 1** — ne pas utiliser pour la présentation avant ce comptage.

---

### 2026-10-05 — Suppression du squelette de notebook prématuré

**Phase:**  
Préparation de la structure du dépôt, avant la Partie 1.

**Goal:**  
Garder le notebook vide de code jusqu'à la construction incrémentale avec les étudiants.

**Owner:**  
Implementation Worker (suppression) ; Copilot (revue et documentation).

**What we did:**  
Supprimé le squelette de notebook prérempli sans créer de nouveau notebook ni modifier le code de dataset ou de modèle ; mis à jour README et PROJECT_STATE.

**Why:**  
Le squelette contenait déjà du comptage d'images et des étapes ultérieures, avant l'exploration d'arborescence prévue en 1a.

**Important code/concept:**  
Construire le notebook progressivement ; la micro-étape 1a se limite à explorer l'arborescence.

**Result:**  
Le dossier `notebooks/` est vide ; la vérification globale ne trouve plus le squelette ni son symbole de comptage. FER2013 est documenté comme APPROVED et les statistiques restent à mesurer en Partie 1.

**Decision:**  
KEEP — commencer par 1a ; ne pas ajouter de comptage, tableau ou autre cellule dans cette étape.

**Problems encountered:**  
Un squelette avait anticipé des travaux postérieurs ; il a été retiré avant le démarrage de la Partie 1.

**Reviewer:**  
Copilot: PASS.

**Presentation material:**  
- Le notebook sera construit progressivement plutôt que prérempli.
- Les statistiques seront mesurées pendant l'analyse réelle du dataset.

---

### 2026-10-05 — Commit de référence Git

**Phase:**
Préparation du dépôt avant la Partie 1.

**Goal:**
Créer un point de référence propre pour les revues différentielles.

**Owner:**
Tiny Local Worker ; Reviewer : Copilot.

**What we did:**
Créé le commit initial `95fcc08` avec les six fichiers du projet et la règle `.gitignore` `*.pdf`.

**Why:**
Permettre de comparer les prochaines modifications et empêcher l'ajout du PDF de l'énoncé.

**Result:**
Un commit sur `main`, arbre de travail propre ; le PDF est ignoré (code 0), et aucun PDF, ZIP, jeu de données ou identifiant Kaggle n'est suivi. Le contrôle de `.hermes.md` n'a trouvé que des mentions de règles sur secrets/tokens, pas de valeur de secret.

**Decision:**
KEEP.

**Reviewer:**
Copilot: PASS.

**Presentation material:**
- Le commit de référence permet des revues de code basées sur le diff.
- Les données locales et PDF sont exclus du suivi Git.

### 2026-10-05 — FER2013 directory inventory (micro-step 1a)

**Phase:**  
Part 1 — Dataset research and understanding.

**Goal:**  
Verify the extracted dataset layout and per-class image counts before loading data.

**Owner:**  
Student A (Nikko; assigned owner); implementation worker.

**What we did:**  
Added a `pathlib`-only notebook cell that lists sorted class folders and counts `.jpg` files in `train` and `test`.

**Why:**  
Class folders provide labels, and checking the directory layout first keeps the data-loading step grounded in the actual dataset.

**Important code/concept:**  
Notebook cell 1a; folder names as class labels.

**Result:**  
The notebook JSON is valid; the 13-line code cell executed successfully from the project root and reported 28,709 train images and 7,178 test images, with seven class counts in each split.

**Decision:**  
KEEP — proceed to the next Part 1 micro-step only when assigned.

**Problems encountered:**  
The initial count matched every directory entry; it was narrowed to `*.jpg` so the displayed value explicitly counts images. The totals remained unchanged.

**Reviewer:**  
Copilot: PASS.

**Presentation material:**  
- FER2013's directory names map to the seven expression labels.
- The training split is imbalanced: `disgust` has 436 images while `happy` has 7,215; later evaluation should consider class-wise performance.

### 2026-10-05 — Dataset format and input pipeline (1b–1c)

**Phase:**  
Part 1 — Dataset research and understanding.

**Goal:**  
Verify the image format and prepare reproducible train, validation, and test datasets.

**Owner:**  
Implementation Worker (1c); Student B (Simon) will run Colab, Student A (Nikko) will review the results.

**What we did:**  
Added a 1b image grid and full train/test format check, then added 1c dataset loading, one-hot labels, normalization, and image-count reporting. Copilot reviewed the notebook and confirmed that `PROJECT_STATE.md` reflects the current status.

**Why:**  
Confirm the model's input shape and keep validation separate from the final test set.

**Important code/concept:**  
Images use shape `(48, 48, 1)`; `image_dataset_from_directory` creates grayscale batches with categorical labels; named `normaliser` scales pixels by 255.

**Result:**  
The 1b cell ran locally: all 35,887 images are 48×48, PIL mode `L`, `.jpg`, with no exceptions. The 1c cell is valid Python and notebook JSON, but awaits Colab execution because TensorFlow is not installed locally; expected counts are 22,968 / 5,741 / 7,178.

**Decision:**  
CONTINUE — confirm the 1c batch shapes, pixel range, and counts in Colab before proceeding to the dense baseline.

**Problems encountered:**  
The 1c explanation initially overstated normalization as preventing exploding gradients; it was corrected to describe improved training stability and typical speed-up without making that guarantee.

**Reviewer:**  
Copilot: PASS.

**Presentation material:**  
- FER2013 images are already 48×48 grayscale, so the input shape is `(48, 48, 1)`.
- Validation tunes/monitors training; the test split remains reserved for final evaluation.


### 2026-10-06 — Refactor of the notebook and the team workflow

**Phase:**  
Part 1 — Dataset research and preparation.

**Goal:**  
Remove duplication, make the notebook run on Colab, and cut the agent overhead
that was slowing the project down.

**Owner:**  
AI assistant (`ponytail-audit` + `mle-workflow` skills).

**What we did:**  
- Step 0 now defines the shared constants once (`DATA_DIR`, `CLASS_NAMES`,
  `IMAGE_SIZE`, `SEED`), sets a global seed and no longer fails on Colab when
  the repository is not cloned.
- 1a/1b/1c reuse these constants; 1b counts formats with `collections.Counter`;
  1c adds `.cache().prefetch()`.
- Team rules reduced from ~590 to ~70 lines; Codex only at decision points,
  Copilot review once per Part, Qwen optional.
- `PROJECT_CONTEXT.md` merged into `PROJECT_STATE.md`; contradictions fixed
  (remote exists, `part3_cnn.ipynb` never existed).

**Why:**  
Each agent re-read ~8k tokens of rules on every turn and every cell went
through 6–8 agent calls; Part 1 took a full evening with 3 days left.

**Result:**  
Notebook executed end to end on a synthetic dataset with the same folder
layout (format anomaly detected, batch shapes (64, 48, 48, 1) / (64, 7),
pixels in [0, 1]). Real-data run in Colab still pending.

**Decision:**  
KEEP.

**Presentation material:**  
- Fixing the seed makes experiments comparable.
- Keeping the test set untouched until the final evaluation.

### 2026-10-06 — Switch to a single assistant

**Goal:**  
Stop the slow, forgetful group chat.

**What we did:**  
Replaced the five-agent group chat with one assistant that plans, codes,
explains and updates the docs. A second model is consulted by hand only at key
decisions. The team rules moved from `.hermes.md` to `AGENTS.md`, a
tool-neutral file that any assistant can read (`CLAUDE.md` points to it).

**Why:**  
In the group chat every bot takes a turn when nobody is mentioned, and each
bot keeps its own isolated memory, so context was lost between agents and every
message cost several model calls.

**Decision:**  
KEEP. Shared memory across tools to be revisited later if needed; for now the
memory is `PROJECT_STATE.md` + this journal.

### 2026-10-06 — Dense baseline (Part 2) and train shuffling fix

**Phase:**  
Part 2 — Dense baseline.

**Goal:**  
Get a reference score that the CNN must beat.

**Owner:**  
Student A (Nikko); reviewer: Student B (Simon).

**What we did:**  
Added a dense network (Flatten → Dense 256 → Dense 128 → softmax 7, 623,879
parameters), trained it for 20 fixed epochs on train with validation
monitoring, plotted the learning curves and compared the validation accuracy
with two references: chance (1/7 ≈ 14 %) and always predicting the majority
class (≈ 25 % for `happy`).

**Problems encountered:**  
`.cache()` on the training set froze the batch order of the first epoch, so
the images were no longer reshuffled between epochs (checked by iterating the
dataset twice). Fixed by caching only the validation and test sets.

**Result:**  
Runs end to end on a synthetic dataset; validation accuracy on real data to
be measured in Colab.

**Decision:**  
CONTINUE — run in Colab, then Part 3 (CNN).

**Presentation material:**  
- A dense network loses the notion of neighbouring pixels after `Flatten`;
  this motivates the CNN.
- Compare against chance and the majority class, not only against 0 %.

### 2026-10-06 — Notebook aligned with the assignment PDF

**Phase:**  
Parts 1–2.

**Goal:**  
Cover every item the assignment asks for in Parts 1 and 2.

**What we did:**  
- Part 1: dataset description cell (source, authors, licence DbCL v1.0,
  collection method, counts, classes, size, format, imbalance) and a
  class-distribution bar chart in 1a.
- Part 2: baseline reduced to the assignment's scheme (Flatten → Dense 128 ReLU
  → Dense 7 softmax, 295,943 parameters); theory cell on weights and biases,
  forward pass, activations, loss, backpropagation and gradient descent; 2d
  shows the 7 softmax probabilities for one validation image.
- Intro cell now names Nikko and Simon.

**Why:**  
The PDF lists what must be shown and explained; the licence, the authors, the
imbalance chart and the learning theory were missing.

**Decision:**  
KEEP.

**Presentation material:**  
- Class-distribution chart: `disgust` has ≈ 16.5× fewer training images than `happy`.
- Loss example: probability 0.72 on the true class gives L ≈ 0.33; 0.05 gives L ≈ 3.0.

### 2026-10-07 — CNN, first training and evaluation

**Phase:**  
Parts 3–5.

**Goal:**  
Complete Part 3 and get a first confusion matrix and ROC curves for the CNN.

**What we did:**  
- Part 3: CNN from the assignment diagram (Conv 32 → Pool → Conv 64 → Pool →
  Flatten → Dense 7 softmax, 63,623 parameters), theory cell, layer-by-layer
  table, justification.
- 4a: training with early stopping on `val_loss` (patience 3, best weights restored).
- 5a: confusion matrix and per-class recall; 5b: one-vs-rest ROC curves with AUC.
  Both on the validation set; the test set is untouched.

**Why:**  
The confusion matrix and the ROC curves need a trained model, so a minimal
training cell was added before them. Early stopping makes the evaluated
weights those of the best epoch instead of an overfitted one.

**Result:**  
Single local CPU run, seed 42, validation: baseline 36.1 %, CNN 49.7 %
(stopped after 10 epochs, best epoch 7). Recall from 9.6 % (`disgust`) to
72.0 % (`happy`).

**Decision:**  
KEEP — finish Parts 4 and 5, then experiments.

**Presentation material:**  
- The CNN beats the dense baseline by 13.6 points with 4.6× fewer parameters.
- `disgust`: 7 images out of 73 recognised — accuracy alone hides this.

### 2026-10-07 — Parts 4–6: training analysis, error analysis, experiments, final model

**Phase:**  
Parts 4, 5 and 6.

**Goal:**  
Finish the mandatory parts: justify the training choices, analyse the curves
and the errors, run at least 3 experiments and pick a final model.

**Owner:**  
Both (to review and explain before the defence).

**What we did:**  
- Part 4: table justifying loss, optimiser, batch size, epochs and metrics;
  4b learning curves of the CNN with the best epoch marked.
- Part 5: 5c top confusions and classes sorted by recall; 5d grid of 4
  correct and the 8 most confident wrong predictions.
- 3a now resets the seed just before building the CNN; each experiment does
  the same, so every model starts from the same random state. This moved the
  `cnn_base` numbers (49.7 % → 50.3 %, best epoch 7 of 10).
- Part 6: five experiments, each changing one thing from the last retained
  model, all measured on validation; comparison table; final model E3; one
  test evaluation; model saved to `models/modele_final.keras`; demo cell.

**Why:**  
The assignment asks for curve interpretation, confused classes with reasons,
examples with probabilities including errors, and ≥ 3 controlled experiments.

**Experiments (validation, seed 42, max 30 epochs, early stopping patience 3):**

| # | Hypothesis | Change | Before | After | Decision |
|---|---|---|---|---|---|
| E1 | a hidden layer combines conv features before deciding | + Dense(128) ReLU | cnn_base 50.3 %, loss 1.322 | 49.8 %, loss 1.309, best epoch 5 (overfits earlier) | KEEP (tie < 1 point, lower loss, assignment figure 2) |
| E2 | dropout slows memorisation in the 819k-parameter Dense layer | + Dropout(0.5) after Flatten | 49.8 %, loss 1.309 | 52.3 %, loss 1.235, best epoch 9 | KEEP |
| E3 | a third block sees larger patterns and shrinks the Dense input (6,400 → 2,048) | + Conv2D(128) + MaxPooling | 52.3 %, 839,047 params | 57.7 %, loss 1.126, 355,847 params | KEEP — final model |
| E4 | augmentation reduces the +7.8-point train/val gap | + RandomFlip, RandomRotation(0.05), RandomZoom(0.1) | 57.7 % | 55.4 %, gap −1.7, learns much slower, stopped at epoch 19 | REJECT |
| E5 | class weights raise `disgust` recall | E3 + class_weight (disgust 9.41, happy 0.57) | 57.7 %, disgust 34.2 % | 53.9 %, disgust 57.5 %; happy −8.6, angry −5.2, fear −4.6 points | REJECT for the final model |

**Result:**  
Final model E3 (`e3_3_blocs`). Test set, used once: **59.3 %** accuracy
(always `happy` = 24.7 %); recall angry 48.7 · disgust 33.3 · fear 35.9 ·
happy 84.0 · neutral 58.0 · sad 47.8 · surprise 70.5 %. Full notebook run:
543 s on CPU (Windows, TensorFlow 2.21), identical numbers on two runs.
Not yet reproduced on Colab.

**Decision:**  
KEEP E3 as final model. CONTINUE: Colab check, student review, slides.

**Problems encountered:**  
- A quick standalone script was used before writing the cells to time the
  runs and try the experiment order. Augmentation added on the two-block
  model (with dropout) gave 51.7 % vs 52.8 % without (train accuracy below
  validation: the model was too small), so the third conv block was tested
  before augmentation. These script numbers are not in the notebook.
- E5 was first chained after E4; since E4 was rejected, E5 was rebuilt on
  E3 to respect "one change from the last retained model".
- The test cells were left out of the first two runs, so the final model was
  chosen from validation only before the test set was evaluated.

**Reviewer:**  
Pending (students).

**Presentation material:**  
- Deeper convolutions beat a bigger Dense layer: E3 has 2.4× fewer
  parameters than E2 and +5.4 points.
- Class weights trade accuracy for `disgust` recall (34 % → 58 %, −3.8 points
  of accuracy): the right choice depends on what the application needs.
- Several "errors" in 5d look like wrong labels (smiling faces labelled `sad`
  or `surprise`): FER2013 label noise limits any model.

### 2026-10-08 — Remove blank images, student-style notebook

**What we did:**
- Found 13 blank images in FER2013 (pixel std < 1, no face): 12 in train
  (7 in `angry`), 1 in test. Step 0 now deletes them; the dataset table
  mentions it.
- Rewrote the notebook text and comments in a shorter student style; long
  theory moved to `soutenance/notebook_version_detaillee.ipynb`.
- Re-ran the whole notebook (local CPU) and updated every number.

**Why:**
Blank images only teach noise. Removing them changes the train/validation
split slightly, so all experiments were re-run.

**Result:**
Same decisions, same final model (E3). Validation: base CNN 50.8 %, E1 51.9,
E2 52.2, E3 58.3, E4 56.6, E5 53.8. Test (once): 57.7 % on 7,177 images.
E1 is now kept on accuracy (+1.1); E2 on loss and train/val gap (3.6 vs 7.9).

**Decision:**
KEEP.
