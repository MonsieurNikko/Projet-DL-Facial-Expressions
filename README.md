# Reconnaissance d'expressions faciales — FER2013

Modèle de deep learning (Keras / TensorFlow) qui classe un visage en niveaux de
gris 48×48 parmi 7 expressions : `angry`, `disgust`, `fear`, `happy`,
`neutral`, `sad`, `surprise`.

Projet de cours — Nikko et Simon.

## Démarrage rapide (Google Colab)

1. Ouvrir le notebook dans Colab :
   [notebooks/fer2013_expressions.ipynb](https://colab.research.google.com/github/MonsieurNikko/Projet-DL-Facial-Expressions/blob/main/notebooks/fer2013_expressions.ipynb)
   Le dépôt est privé : la première fois, Colab affiche « Notebook introuvable ».
   Cliquer sur **Autoriser avec GitHub** et cocher l'accès aux dépôts privés
   (chaque personne doit être collaboratrice du dépôt).
2. Activer le GPU : **Exécution → Modifier le type d'exécution → GPU**.
3. Lancer les cellules dans l'ordre (**Exécution → Tout exécuter**).

La première cellule télécharge FER2013 automatiquement (≈ 200 Mo) si les
données ne sont pas déjà présentes. Aucun compte ni clé Kaggle n'est demandé.

## Installation locale

Prérequis : Python 3.10 ou plus récent.

```bash
git clone https://github.com/MonsieurNikko/Projet-DL-Facial-Expressions.git
cd Projet-DL-Facial-Expressions
pip install tensorflow matplotlib pillow scikit-learn kagglehub jupyter
jupyter notebook notebooks/fer2013_expressions.ipynb
```

Sans GPU, l'entraînement est nettement plus lent : Colab est recommandé.

## Dataset

[FER2013 sur Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
(`msambare/fer2013`) — 35 887 images 48×48 en niveaux de gris, une classe par dossier.
Licence : Database Contents License (DbCL) v1.0. Auteurs : Pierre-Luc Carrier et
Aaron Courville ([Goodfellow et al., 2013](https://arxiv.org/abs/1307.0414)).

| Classe   | Train | Test  |
|----------|------:|------:|
| angry    | 3 995 |   958 |
| disgust  |   436 |   111 |
| fear     | 4 097 | 1 024 |
| happy    | 7 215 | 1 774 |
| neutral  | 4 965 | 1 233 |
| sad      | 4 830 | 1 247 |
| surprise | 3 171 |   831 |
| **Total** | **28 709** | **7 178** |

Découpage utilisé : 80 % du dossier `train` pour l'entraînement (22 968),
20 % pour la validation (5 741, graine 42), et le dossier `test` (7 178)
uniquement pour l'évaluation finale.

Les classes sont déséquilibrées (`disgust` a environ 16 fois moins d'images que
`happy`) : la précision globale seule ne suffit pas, on regarde aussi la
matrice de confusion et le rappel par classe.

## Contenu du notebook

| Partie | Contenu |
|--------|---------|
| 0 | Configuration commune, graine aléatoire, téléchargement des données |
| 1 | Exploration du dataset, vérification du format, datasets train / validation / test |
| 2 | Baseline dense |
| 3 | Réseau convolutif (CNN) |
| 4 | Entraînement |
| 5 | Évaluation et analyse des erreurs |
| 6 | Expériences comparées sur la validation |

## Résultats

À compléter après l'évaluation finale sur le jeu de test.

## Structure

```
.
├── notebooks/
│   └── fer2013_expressions.ipynb   # le notebook du projet
├── data/                           # dataset téléchargé (non versionné)
├── models/                         # modèles entraînés (non versionnés)
└── docs/                           # énoncé du projet (non versionné)
```

## Problèmes fréquents

- **`ModuleNotFoundError: kagglehub`** en local : `pip install kagglehub`.
- **Entraînement très lent sur Colab** : vérifier que le GPU est activé (étape 2 du démarrage rapide).
- **Données corrompues ou incomplètes** : supprimer le dossier `data/fer2013/` et relancer la première cellule.
