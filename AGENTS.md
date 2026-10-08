# AGENTS.md — Rules for any AI coding assistant on this project

These rules apply to every assistant (Claude Code, Codex, Copilot, Cursor,
…). Any of them must be able to pick up the project from the files alone.
Keep this file short: it is read on every session.

## Start of every session

1. Read this file.
2. Read `PROJECT_STATE.md` (current truth).
3. Read `PROJECT_JOURNAL.md` only when the history matters.

Never ask the students for something already written in these files.
At the end of each work cycle, update `PROJECT_STATE.md`, and the journal
for milestones, so the next assistant can continue.

## Priorities

1. Current explicit human instruction.
2. Official assignment PDF (repo root, `Projet_DL_Expressions_faciales (1).pdf`,
   not versioned). Read it before planning any part.
3. This file.
4. `PROJECT_STATE.md`, then `PROJECT_JOURNAL.md`.

Learning first, correctness second, speed third. Deadline: 2026-10-09.
Mandatory Parts 1–6 before any optional extension.

## Work cycle

One cycle = one coherent step (e.g. "dense baseline", "training curves"),
usually 1–3 notebook cells with their Markdown explanation.

1. State WHAT / WHY / OWNER (Nikko or Simon, alternate each step).
2. Write the cells. An assistant may run them locally (`.venv`, CPU only on
   Windows) to get real numbers; the students then re-run the whole notebook
   in Colab with the GPU before the defence.
3. Explain the key lines and ask one comprehension question.
4. Update `PROJECT_STATE.md` (and the journal for milestones).
5. Stop and wait for the students' `go`.

At key decisions only (CNN architecture, evaluation method, final model
choice), write a short summary the students can give to another model for
a one-round critique.

## Code rules

- Beginner-readable: explicit names, short cells, simple loops, no clever
  tricks. Ask: "Could Nikko or Simon explain this line by line at the oral?"
- Shared constants (`DATA_DIR`, `CLASS_NAMES`, `IMAGE_SIZE`, `SEED`) are
  defined once in step 0; never redefine them.
- Comments explain WHY, not syntax.
- No shortcuts the students cannot explain: no nested comprehensions, no
  lambdas, no dense one-liners. A plain `for` loop is better.
- Code and comments stay neutral: never mention an AI assistant or tool by
  name. The AI tools used are disclosed openly in the README ("Outils
  utilisés") and the journal; do not hide or deny them.

## Writing style (notebook, README)

Write like a student who saves time, not like a polished tutorial:
- Markdown: short sentences with "on", a few lines per cell. Almost no bold,
  no arrows (`→`), no long dashes (`—`), few bullet lists, no bold labels.
- Comments: few, only where a choice is not obvious. Not aligned in columns.
- Prints: simple, e.g. `print("tailles:", dict(tailles))`. No padded
  f-strings or column alignment.
- Keep everything the assignment asks for (each required point keeps at
  least one sentence; tables required by the PDF stay).
- Numbers from a training run (accuracies, losses, best epochs, recalls,
  confusion counts) change from one run to another. In the interpretation
  cells, describe the trend ("plus gros gain", "surapprend plus tôt"); the
  code prints the exact values. Measured numbers appear only in the dated
  reference tables (6g and 6i), always with the date and the machine of the
  run, and in `PROJECT_STATE.md` / the journal. Fixed facts (dataset counts,
  parameter counts, hyperparameters) can stay anywhere.
- One notebook only (`notebooks/fer2013_expressions.ipynb`). Never create a
  second copy of the code: it drifts. Long theory (backpropagation, padding,
  stride…) and the jury questions go to `soutenance/explications.md` (text,
  no executable code), not in the notebook.
- Run the `humanizer` skill on new prose.

## Defence

- Notebook first; presentation format (slides or notebook) still to decide.
- Prepare likely questions (why this data, why this logic, why this code)
  with answers in `soutenance/`, not in the notebook.
- Both students must be able to explain every cell.

## Environment

- Local: `.venv` (Python 3.11, TensorFlow CPU). TensorFlow has no GPU support
  on native Windows since 2.11; use Colab (GPU T4) or WSL2
  (`tensorflow[and-cuda]`) for speed.

## README

The README is for people who want to use the project, not a story of how it
was built. Keep only: what the project does, how to run it (Colab and local),
the dataset, the notebook contents, the results, the tools used (including
the AI assistants and skills), the structure and common problems. No
development history, workflow details or decisions: those belong in
`PROJECT_JOURNAL.md`. Update the results section once measured.

## Deep-learning rules

- Validation for tuning and experiments; the test set is used **once**,
  for the final model only.
- Classes are imbalanced: report a confusion matrix and per-class recall,
  not only accuracy.
- Experiments: change one variable at a time, same seed, record
  hypothesis / change / before / after / decision in the journal.
- Never claim an improvement without a measured number.

## Git

- Commit messages: Conventional Commits, concrete and neutral, e.g.
  `feat(part2): add dense baseline model` or
  `fix(part1): keep data download working on Colab`.
  No assistant or tool names, no co-author or session trailers.
- Never commit `data/`, `models/`, `kaggle.json`, `.env` or the PDF.
- Ask the humans before pushing.

## Skills

Reusable instructions live in `.claude/skills/<name>/SKILL.md` (plain
Markdown, readable by any assistant): `ponytail` (simplest solution),
`ponytail-review`, `ponytail-audit`, `mle-workflow` (ML method),
`scientific-thinking-literature-review`, `scientific-thinking-scholar-evaluation`.
`i-have-adhd` (action-first answers for Nikko: next action first, numbered
steps, state restated each turn; on with `/i-have-adhd`, off with "stop adhd mode").
Also used from the user's global setup: `humanizer` (natural prose),
`graphify` (repo analysis) and the ECC skills and review agents. Use the
relevant skills when the students ask for it.
