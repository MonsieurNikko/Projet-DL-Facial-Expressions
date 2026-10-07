# Reconnaissance d'expressions faciales (FER2013)

Modèle de deep learning (Keras / TensorFlow) qui classe un visage en niveaux de
gris 48x48 parmi 7 expressions : `angry`, `disgust`, `fear`, `happy`,
`neutral`, `sad`, `surprise`.

Projet de cours de Nikko et Simon.

## Lancer le projet sur Google Colab

1. Ouvrir le notebook dans Colab :
   [notebooks/fer2013_expressions.ipynb](https://colab.research.google.com/github/MonsieurNikko/Projet-DL-Facial-Expressions/blob/main/notebooks/fer2013_expressions.ipynb).
   Le dépôt est privé, donc la première fois Colab affiche "Notebook introuvable".
   Il faut cliquer sur "Autoriser avec GitHub" et cocher l'accès aux dépôts
   privés (chaque personne doit être collaboratrice du dépôt).
2. Activer le GPU : menu Exécution, Modifier le type d'exécution, GPU.
3. Lancer toutes les cellules dans l'ordre (Exécution, Tout exécuter).

La première cellule télécharge FER2013 (environ 200 Mo) si les données ne sont
pas déjà là. Pas besoin de compte ni de clé Kaggle.

## Installation locale

Il faut Python 3.10 ou plus récent.

```bash
git clone https://github.com/MonsieurNikko/Projet-DL-Facial-Expressions.git
cd Projet-DL-Facial-Expressions
pip install tensorflow matplotlib pillow scikit-learn kagglehub jupyter
jupyter notebook notebooks/fer2013_expressions.ipynb
```

Sans GPU l'entraînement est beaucoup plus lent (environ 9 minutes pour tout le
notebook sur CPU, contre quelques minutes sur Colab).

## Dataset

[FER2013 sur Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
(`msambare/fer2013`) : 35 887 images 48x48 en niveaux de gris, un dossier par
classe. Licence Database Contents License (DbCL) v1.0. Auteurs : Pierre-Luc
Carrier et Aaron Courville ([Goodfellow et al., 2013](https://arxiv.org/abs/1307.0414)).

| Classe | Train | Test |
|---|---:|---:|
| angry | 3 995 | 958 |
| disgust | 436 | 111 |
| fear | 4 097 | 1 024 |
| happy | 7 215 | 1 774 |
| neutral | 4 965 | 1 233 |
| sad | 4 830 | 1 247 |
| surprise | 3 171 | 831 |
| Total | 28 709 | 7 178 |

Découpage : 80 % du dossier `train` pour l'entraînement (22 968 images), 20 %
pour la validation (5 741, seed 42), et le dossier `test` (7 178) seulement
pour l'évaluation finale.

Les classes sont déséquilibrées (`disgust` a environ 16 fois moins d'images que
`happy`). L'accuracy seule ne suffit donc pas, on regarde aussi la matrice de
confusion et le rappel par classe.

## Contenu du notebook

| Partie | Contenu |
|---|---|
| 0 | Configuration, seed, téléchargement des données |
| 1 | Exploration du dataset, vérification du format, datasets train / validation / test |
| 2 | Baseline dense |
| 3 | CNN |
| 4 | Entraînement |
| 5 | Évaluation et analyse des erreurs |
| 6 | Expériences comparées sur la validation, test final, démo |

## Résultats

Chiffres d'une exécution locale complète du notebook (CPU, Windows,
TensorFlow 2.21, seed 42). Ils sont à revérifier sur Colab, le GPU peut donner
des valeurs un peu différentes.

Comparaison des modèles sur la validation (5 741 images) :

| Modèle | Changement | Accuracy | Rappel `disgust` |
|---|---|---:|---:|
| Baseline dense | Flatten, Dense 128, softmax | 36,1 % | |
| CNN de base | 2 blocs Conv + MaxPooling, softmax | 50,3 % | 13,7 % |
| E1 | + couche `Dense(128)` | 49,8 % | 11,0 % |
| E2 | + `Dropout(0.5)` | 52,3 % | 17,8 % |
| E3 (modèle final) | + 3e bloc `Conv2D(128)` | 57,7 % | 34,2 % |
| E4 | E3 + augmentation de données | 55,4 % | 6,8 % |
| E5 | E3 + pondération des classes | 53,9 % | 57,5 % |

Repères : hasard 14,3 %, répondre toujours `happy` 24,4 %.

Le modèle final (E3, 355 847 paramètres) fait 59,3 % d'accuracy sur le jeu de
test (7 178 images, utilisé une seule fois).

| Classe | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rappel (test) | 48,7 % | 33,3 % | 35,9 % | 84,0 % | 58,0 % | 47,8 % | 70,5 % |

Le notebook sauvegarde le modèle dans `models/modele_final.keras`. La dernière
cellule (Démo) le recharge et prédit l'expression d'une image.

## Outils utilisés

Technologies :
- Python, TensorFlow / Keras pour les modèles
- NumPy, scikit-learn (matrice de confusion, courbes ROC), Matplotlib, Pillow
- kagglehub pour télécharger le dataset
- Jupyter et Google Colab pour exécuter le notebook, VS Code en local
- Git et GitHub

Assistants d'IA : on s'est servi de plusieurs assistants pendant le projet,
pour écrire une partie du code et des explications, relire le travail, lancer
les expériences de la partie 6 et organiser la documentation.
- Claude Code avec des skills : `ponytail`, `ponytail-review` et
  `ponytail-audit` (garder le code simple), `mle-workflow` (méthode ML,
  reproductibilité), `scientific-thinking-literature-review` et
  `scientific-thinking-scholar-evaluation`, `graphify` (analyse du dépôt), et les skills et agents ECC pour la
  relecture. On a aussi utilisé des sous-agents Claude Code : un pour analyser
  le dépôt, un autre pour écrire les parties 4 à 6 et lancer les expériences
  en local.

Les skills du projet sont dans `.claude/skills/`, et les règles données aux
assistants dans `AGENTS.md`. Tous les chiffres viennent d'exécutions réelles du
notebook.

## Structure

```
.
├── notebooks/
│   └── fer2013_expressions.ipynb      # le notebook du projet
├── soutenance/                        # préparation de l'oral (version détaillée du notebook)
├── .claude/skills/                    # skills utilisés par les assistants d'IA
├── AGENTS.md                          # règles de travail pour les assistants
├── PROJECT_STATE.md                   # état actuel du projet
├── PROJECT_JOURNAL.md                 # historique des étapes et des décisions
├── data/                              # dataset téléchargé (non versionné)
└── models/                            # modèles entraînés (non versionnés)
```

L'énoncé du projet (PDF) n'est pas versionné.

## Problèmes fréquents

- `ModuleNotFoundError: kagglehub` en local : `pip install kagglehub`.
- Entraînement très lent sur Colab : vérifier que le GPU est activé (étape 2 plus haut).
- Entraînement lent en local sous Windows : TensorFlow n'utilise pas le GPU sous
  Windows depuis la version 2.11, même avec une carte NVIDIA. Utiliser Colab, ou
  WSL2 avec `pip install tensorflow[and-cuda]`.
- Données corrompues ou incomplètes : supprimer `data/fer2013/` et relancer la
  première cellule.
