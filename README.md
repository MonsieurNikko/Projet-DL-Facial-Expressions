# Projet DL : Reconnaissance d'expressions faciales (FER2013)

## Objectif
Développer un modèle de deep learning qui reconnaît les 7 expressions faciales du dataset FER2013.

## Lancer le notebook
Ouvrir `notebooks/fer2013_expressions.ipynb` dans Google Colab (GPU activé) et exécuter les cellules dans l'ordre.
L'étape 0 télécharge automatiquement FER2013 dans `data/fer2013/` s'il manque (`kagglehub`, déjà installé sur Colab ; `pip install kagglehub` en local).

## Structure
```
.
├── .claude/skills/      # skills Claude Code (Ponytail, ECC)
├── .hermes.md           # règles de l'équipe d'agents
├── PROJECT_STATE.md     # état actuel du projet
├── PROJECT_JOURNAL.md   # historique validé (pour l'oral)
├── notebooks/
│   └── fer2013_expressions.ipynb
├── docs/                # énoncé PDF (non versionné)
└── data/                # dataset local (non versionné)
```

## Règles
- Ne jamais committer `data/`, `models/` ou `kaggle.json`.
