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
2. Official assignment PDF (`docs/`).
3. This file.
4. `PROJECT_STATE.md`, then `PROJECT_JOURNAL.md`.

Learning first, correctness second, speed third. Deadline: 2026-10-09.
Mandatory Parts 1–6 before any optional extension.

## Work cycle

One cycle = one coherent step (e.g. "dense baseline", "training curves"),
usually 1–3 notebook cells with their Markdown explanation.

1. State WHAT / WHY / OWNER (Nikko or Simon, alternate each step).
2. Write the cells; the students **run them in Colab** and share the outputs.
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
- Code, comments and docs stay neutral: never mention an AI assistant or
  tool by name.

## README

The README is for people who want to use the project, not a story of how it
was built. Keep only: what the project does, how to run it (Colab and local),
the dataset, the notebook contents, the results, the structure and common
problems. No development history, workflow, agents or decisions — those
belong in `PROJECT_JOURNAL.md`. Update the results section once measured.

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
