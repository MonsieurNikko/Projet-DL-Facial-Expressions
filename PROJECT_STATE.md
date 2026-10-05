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

Part 1 / micro-step 1a — explore the directory tree only (not started).
Owner: Student A; Reviewer: Student B.
Blocked by the prerequisites listed in Open issues.


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
à mesurer en Partie 1

Classes:
à mesurer en Partie 1 (FER2013 a 7 classes : angry, disgust, fear, happy, neutral, sad, surprise)

Image dimensions:
à mesurer en Partie 1

Class balance:
à mesurer en Partie 1


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

- Local Git repository initialized on branch `main`; no commits and no remote.
- Git tracks no files; all non-ignored repository files are untracked.
- `notebooks/` is empty.
- No `data/` directory or local `~/.kaggle/kaggle.json`; FER2013 has not been downloaded.
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

- Dataset statistics, class balance, dimensions, and license details remain to be verified in Part 1.
- No reference commit exists, so reviews cannot yet rely on `git diff`.
- Part 1a prerequisites are unmet: the dataset is unavailable (no `data/` and no Kaggle credentials). The Colab download method has not been decided.
- The rules were refactored: CODEX_GATE is mandatory before starting 1a.


## Next candidate step

No further implementation step is approved; reassess after micro-step 1a is completed and reviewed.

Micro-step 1a scope: explore the directory tree only; no image counting, table, or other notebook cell.