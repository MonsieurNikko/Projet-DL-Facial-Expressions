# Explications pour la soutenance

Ce fichier accompagne `notebooks/fer2013_expressions.ipynb`, le seul notebook du projet. Il ne contient pas de code exécutable : il garde les explications longues (théorie, justification des choix) et les questions probables du jury avec leurs réponses. Les sections suivent l'ordre du notebook. Les résultats chiffrés sont dans le notebook (tableaux 6g et 6i, datés) et dans `PROJECT_STATE.md`.

Sommaire :
1. Explications du notebook, partie par partie
2. Questions du jury sur la Partie 1, avec les réponses

---

# 1. Explications du notebook
### Étape 0 — Configuration commune et téléchargement des données

Cette cellule est lancée **en premier** ; toutes les suivantes en dépendent.

1. **Racine du projet.** En local, c'est le dossier qui contient `.gitignore` : le notebook marche qu'on le lance depuis la racine ou depuis `notebooks/`. Sur Colab, si le dépôt n'a pas été cloné, il n'y a pas de `.gitignore` : on reste alors dans le dossier courant (`/content`).
2. **Constantes partagées.** `DATA_DIR`, `CLASS_NAMES`, `IMAGE_SIZE` et `SEED` sont définies **une seule fois ici** et réutilisées partout : changer une valeur à un endroit la change pour tout le notebook.
3. **Graine aléatoire (*seed*).** `keras.utils.set_random_seed(42)` fixe le hasard de Python, NumPy et TensorFlow (découpe train/validation, poids initiaux, mélange des lots). Deux expériences ne diffèrent alors que par ce qu'on a volontairement changé — indispensable pour comparer nos ≥ 3 expériences. (Sur GPU, quelques opérations restent légèrement non déterministes : de petites variations restent possibles.)
4. **Téléchargement idempotent.** `data/` n'est pas dans Git (≈ 200 Mo). Si `data/fer2013/train` et `data/fer2013/test` existent déjà, la cellule ne télécharge rien ; sinon elle récupère FER2013 avec `kagglehub` (bibliothèque officielle de Kaggle). La relancer 10 fois donne le même résultat qu'une seule fois.
5. **Images vides.** La cellule supprime ensuite les images presque unies (écart-type des pixels inférieur à 1, donc sans visage) : 13 images, 12 en `train` et 1 en `test`. Leur étiquette est forcément fausse. `unlink()` efface le fichier du disque : pour le récupérer, il faut supprimer `data/fer2013` et relancer l'étape 0.

## Partie 1 — Recherche, compréhension et préparation des données

### Description du dataset : FER2013

| Élément | Valeur |
|---|---|
| **Source** | Kaggle, [`msambare/fer2013`](https://www.kaggle.com/datasets/msambare/fer2013) (version au format images, un dossier par classe) |
| **Auteurs** | Pierre-Luc Carrier et Aaron Courville. Dataset publié pour le concours *Challenges in Representation Learning* (ICML 2013) : Goodfellow et al., 2013, [arXiv:1307.0414](https://arxiv.org/abs/1307.0414) |
| **Licence** | Database Contents License (DbCL) v1.0, indiquée sur la page Kaggle. Les photos d'origine viennent du web : on utilise le dataset dans un cadre académique |
| **Collecte** | Images trouvées via Google Images avec des mots-clés d'émotions ; visages détectés et recadrés automatiquement, vérifiés par des annotateurs, puis réduits en 48 × 48 niveaux de gris |
| **Nombre d'images** | 35 887 : 28 709 en `train`, 7 178 en `test` |
| **Classes** | 7 : `angry` (colère), `disgust` (dégoût), `fear` (peur), `happy` (joie), `neutral` (neutre), `sad` (tristesse), `surprise` |
| **Dimensions** | 48 × 48 pixels |
| **Format** | Niveaux de gris (1 canal), fichiers `.jpg` |
| **Déséquilibre** | Oui : 436 images `disgust` contre 7 215 `happy` en `train` (≈ 16,5 fois moins) |

**Pourquoi ce dataset ?** Il est public, très utilisé comme référence, et ses visages sont **déjà extraits et recadrés** : c'est exactement l'entrée « visage déjà extrait → CNN → expression » des Parties 1 à 6.

Les chiffres du tableau sont mesurés par les cellules 1a et 1b ci-dessous.

### Étape 1a — Compter les images et visualiser le déséquilibre

Avant de charger les données, on compte les images de chaque dossier : chaque dossier de classe sert d'étiquette (label), c'est cette structure que Keras lira. Le graphique montre le **déséquilibre entre les classes** : un modèle aura tendance à mieux reconnaître les classes fréquentes, c'est pourquoi on regardera le rappel par classe en Partie 5, pas seulement l'accuracy.

### Étape 1b — Regarder les images et vérifier le format

**Pourquoi ?** Un modèle apprend à partir de pixels : s'il reçoit des images de tailles ou de modes différents, la couche `Flatten` produira des vecteurs de longueurs différentes et l'entraînement échouera (ou buggera silencieusement).

On fait donc deux choses :
1. **On affiche une grille de 7 classes × 5 images** tirées de `data/fer2013/train/<classe>/` : on vérifie *à l'œil* que les dossiers contiennent bien des visages et que les classes sont cohérentes (utile aussi pour l'oral : montrer des exemples réels).
2. **On vérifie le format sur *toutes* les images** (train + test) avec PIL : taille `(48, 48)`, mode `L` (niveaux de gris) et extension `.jpg`. Un `Counter` (bibliothèque standard Python) compte combien de fois chaque valeur apparaît ; on liste les exceptions plutôt que de les cacher.

### Étape 1c — Construire les datasets d'entraînement, de validation et de test

**Pourquoi ?** Jusqu'ici on a seulement *regardé* les fichiers. Le modèle a besoin d'un `tf.data.Dataset` : un flux de lots d'images + d'étiquettes, prêt pour `model.fit()`.

Cinq points à comprendre :

1. **Normalisation** — les pixels sont des entiers entre 0 et 255. On les divise par 255 pour les ramener dans `[0, 1]` : des entrées petites et à la même échelle rendent l'entraînement plus stable et, en général, plus rapide.
2. **Encodage one-hot** — `label_mode="categorical"` transforme l'étiquette `"happy"` en vecteur `[0, 0, 0, 1, 0, 0, 0]` de 7 valeurs (la position du 1 suit l'ordre de `CLASS_NAMES`). C'est la forme attendue par une sortie `softmax` avec la perte `categorical_crossentropy`.
3. **Validation** — 20 % de `data/fer2013/train` est mis de côté (5 739 images, après la suppression des images vides) pour mesurer la généralisation pendant l'entraînement et repérer le surapprentissage. Ces images n'entraînent jamais les poids : elles servent seulement à décider quand s'arrêter (`early stopping`, choix du meilleur modèle).
4. **Test** — `data/fer2013/test` (7 177 images, après la suppression d'une image vide) reste intact, sans mélange ni découpe : il est réservé à l'évaluation finale, une seule fois, à la toute fin du projet. Si on l'utilisait pendant le développement, on « tricherait » sans le savoir.

Rappel de l'étape 1b : toutes les images font déjà 48×48 en niveaux de gris, donc la forme d'une image est `(48, 48, 1)` et aucune redimensionnement n'est nécessaire.

5. **Performance** — `.prefetch()` prépare le lot suivant pendant que le modèle calcule sur le lot courant. `.cache()` garde les lots en mémoire après la première lecture : on l'utilise **seulement pour la validation et le test**. Sur l'entraînement, il figerait l'ordre des images de la première époque ; or on veut que les images soient **mélangées différemment à chaque époque**, pour que le modèle ne s'habitue pas à un ordre fixe.

## Partie 2 — Baseline dense

**Pourquoi une baseline ?** Avant de construire un CNN, on entraîne le modèle le plus simple possible. Son score sert de **point de comparaison** : si le CNN ne fait pas nettement mieux, c'est qu'il y a un problème. Sans baseline, impossible de dire si 55 % est un bon ou un mauvais résultat.

**Le modèle : un réseau entièrement connecté (*dense*)**, selon le schéma de l'énoncé `Image → Flatten → Dense + activation → Dense → Softmax`.
1. `Flatten` met les 48 × 48 = **2 304 pixels** de l'image bout à bout, dans un seul vecteur.
2. Une couche cachée `Dense(128, activation="relu")` : chacun des 128 neurones est relié à **toutes** les 2 304 valeurs d'entrée. L'activation `relu` (`max(0, x)`) permet au réseau d'apprendre des relations non linéaires.
3. Une couche de sortie `Dense(7, activation="softmax")` donne **7 probabilités qui somment à 1**, une par expression. La classe prédite est celle qui a la plus grande probabilité.

**Sa limite (attendue) :** après `Flatten`, le réseau ne sait plus quels pixels sont voisins. Un sourire décalé de quelques pixels devient un vecteur complètement différent. C'est exactement ce que le CNN de la Partie 3 corrigera.

**Compilation :**
- `loss="categorical_crossentropy"` : la perte adaptée à des étiquettes one-hot et une sortie softmax (étape 1c). Elle est faible quand le modèle donne une forte probabilité à la bonne classe.
- `optimizer="adam"` : l'algorithme qui ajuste les poids pour faire baisser la perte (une version améliorée de la descente de gradient).
- `metrics=["accuracy"]` : le **taux de bonnes réponses**, plus facile à lire que la perte.

### Comment le réseau apprend : poids, propagation avant, perte, rétropropagation

**1. L'entrée.** L'image 48 × 48 devient, après `Flatten`, un vecteur $x$ de 2 304 nombres entre 0 et 1 (un par pixel).

**2. Poids et biais.** Chaque neurone calcule une somme pondérée de ses entrées :

$$z = w_1 x_1 + w_2 x_2 + \dots + w_{2304}\, x_{2304} + b$$

Les **poids** $w$ disent quelle importance donner à chaque pixel ; le **biais** $b$ décale le résultat. Ce sont les seuls nombres que le réseau apprend : ici 295 040 + 903 = **295 943 paramètres**.

**3. Propagation avant (*forward pass*).** L'image traverse le réseau couche par couche :
image → `Flatten` → 128 neurones + ReLU → 7 neurones → softmax → 7 probabilités.

**4. Fonctions d'activation.**
- **ReLU** dans la couche cachée : $\text{ReLU}(z) = \max(0, z)$. Sans fonction non linéaire, empiler des couches reviendrait à une seule couche linéaire, incapable d'apprendre des formes complexes.
- **Softmax** en sortie : $\text{softmax}(z_i) = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ transforme les 7 scores en probabilités positives qui somment à 1. La classe prédite est celle de plus grande probabilité.

**5. La fonction de perte.** La *categorical cross-entropy* vaut $L = -\log(p_{\text{vraie classe}})$ : si le modèle donne 0,72 à la bonne classe, $L \approx 0{,}33$ ; s'il ne lui donne que 0,05, $L \approx 3{,}0$. Plus la perte est basse, meilleur est le modèle.

**6. Rétropropagation (*backpropagation*).** Après chaque lot, on calcule pour **chaque poids** le gradient $\frac{\partial L}{\partial w}$ : de combien la perte change si on modifie un peu ce poids. Le calcul part de la sortie et remonte vers l'entrée, couche par couche, avec la règle de dérivation en chaîne.

**7. Descente de gradient.** Chaque poids est déplacé dans la direction qui fait baisser la perte :

$$w \leftarrow w - \eta \, \frac{\partial L}{\partial w}$$

$\eta$ est le **taux d'apprentissage** (*learning rate*, 0,001 par défaut avec Adam). On fait une mise à jour par lot de 128 images, soit 180 mises à jour par époque (22 958 / 128, le dernier lot n'a que 46 images). **Adam** est une variante de la descente de gradient qui adapte automatiquement le pas de chaque poids.

**8. Sortie multiclasse.** 7 classes → 7 neurones de sortie + softmax (pour 2 classes, on aurait 1 neurone + sigmoïde).

### 2b — Entraîner la baseline

`model.fit()` répète l'apprentissage pendant `EPOCHS_BASELINE` **époques** (une époque = le modèle a vu une fois toutes les images d'entraînement, lot par lot).

À la fin de chaque époque, Keras affiche :
- `loss` / `accuracy` : mesurées sur les images d'**entraînement** ;
- `val_loss` / `val_accuracy` : mesurées sur la **validation**, des images que le modèle n'apprend jamais.

Ce qu'on surveille : si `accuracy` continue de monter alors que `val_accuracy` stagne ou baisse, le modèle **apprend par cœur** les images d'entraînement au lieu de généraliser : c'est le **surapprentissage** (*overfitting*). On garde ici un nombre d'époques fixe pour bien voir ce phénomène sur les courbes ; l'arrêt automatique (*early stopping*) viendra en Partie 4.

### 2c — Courbes d'apprentissage et score de la baseline

On trace la perte et l'accuracy époque par époque, pour l'entraînement et la validation, puis on donne le score final sur la **validation**.

Deux repères pour juger ce score :
- **le hasard** : répondre une classe au hasard donne 1 chance sur 7, soit environ 14 % ;
- **la classe majoritaire** : répondre toujours `happy` (la classe la plus fréquente) donne environ 25 %. Un modèle qui ne dépasse pas ce chiffre n'a rien appris d'utile, même si 25 % paraît mieux que le hasard.

### 2d — Ce que le réseau renvoie pour une image

Pour une image de validation, on affiche les **7 probabilités** produites par softmax, leur somme (= 1) et la classe prédite (la plus probable), à comparer avec la vraie classe.

## Partie 3 — CNN

**Le modèle : un petit réseau convolutif**, selon le schéma de l'énoncé `Image → Conv2D + ReLU → MaxPooling → Conv2D + ReLU → MaxPooling → Flatten → Dense → Softmax`.
1. `Conv2D(32, 5)` fait glisser 32 **filtres** de 5 × 5 pixels sur l'image. Chaque filtre apprend à détecter un petit motif local (un bord, un coin) **où qu'il soit** dans l'image : c'est ce qui manquait à la baseline dense.
2. `MaxPooling2D(2)` ne garde que la plus grande valeur de chaque carré de 2 × 2 : la carte est deux fois plus petite et la détection supporte mieux un petit décalage du visage.
3. Un second bloc `Conv2D(64, 3)` + `MaxPooling2D(2)` combine ces motifs simples en motifs plus grands (un œil, une bouche).
4. `Flatten` puis `Dense(7, activation="softmax")` transforment ces motifs en **7 probabilités**, comme pour la baseline.

La compilation est identique à celle de la baseline (même perte, même optimiseur, même métrique) : les deux modèles seront donc comparables.

### Comment un CNN lit une image : filtres, convolution, pooling

**1. Filtre et taille du noyau (*kernel size*).** Un filtre est une petite grille de poids. La première convolution utilise des filtres de 5 × 5 (`kernel_size=5`), la seconde de 3 × 3. Sur l'image d'entrée, il n'y a qu'un seul canal : un filtre plus large y coûte peu (832 paramètres) et voit un morceau de visage plus grand dès la première couche. Ensuite il y a 32 canaux, un filtre 5 × 5 coûterait presque trois fois plus qu'un 3 × 3 : on reste donc en 3 × 3, la plus petite taille qui a un centre et des voisins de tous les côtés.

**2. Convolution.** Le filtre glisse sur l'image. À chaque position, on multiplie ses poids (25 pour un filtre 5 × 5, 9 pour un 3 × 3) par les pixels qu'il recouvre, on additionne le tout et on ajoute un biais. Les **mêmes poids** servent à toutes les positions : le motif est reconnu où qu'il soit, et il y a très peu de paramètres à apprendre.

**3. Carte de caractéristiques (*feature map*).** Le résultat d'un filtre est une nouvelle « image » qui indique où son motif a été trouvé. 32 filtres donnent 32 cartes, d'où la forme `(44, 44, 32)`. Dans la seconde convolution, chaque filtre regarde les 32 cartes à la fois : il a 3 × 3 × 32 poids.

**4. Pas (*stride*).** C'est le déplacement du filtre entre deux positions. Ici 1 pixel pour les convolutions (valeur par défaut) : on ne saute aucune position.

**5. *Padding*.** Avec `padding="valid"` (valeur par défaut), on n'ajoute rien autour de l'image : le filtre reste à l'intérieur, donc la sortie rétrécit : de 4 pixels avec le filtre 5 × 5 (48 − 5 + 1 = 44), de 2 avec le 3 × 3 (22 − 3 + 1 = 20). Avec `padding="same"`, on entourerait l'image de zéros pour garder la taille. Formule générale :

$$\text{taille de sortie} = \frac{\text{entrée} - \text{noyau} + 2 \times \text{padding}}{\text{pas}} + 1$$

**6. ReLU.** $\max(0, z)$ est appliqué à chaque valeur des cartes : on ne garde que les endroits où le motif est présent, et le réseau devient non linéaire (comme en Partie 2).

**7. *MaxPooling*.** `MaxPooling2D(2)` découpe chaque carte en carrés de 2 × 2 et ne garde que la plus grande valeur : 44 → 22, puis 20 → 10. Il n'a aucun paramètre. Il réduit les calculs et rend la détection moins sensible à un décalage d'un pixel.

**8. `Flatten`, `Dense` et sortie.** Les 64 cartes de 10 × 10 sont mises bout à bout (6 400 valeurs), puis `Dense(7)` + softmax donne les 7 probabilités, comme pour la baseline.

**Tableau couche par couche** (à comparer avec `cnn.summary()` ci-dessus) :

| Couche | Forme de sortie | Paramètres | Calcul |
|---|---|---:|---|
| Entrée | (48, 48, 1) | 0 | |
| `Conv2D(32, 5)` + ReLU | (44, 44, 32) | 832 | (5 × 5 × 1 + 1) × 32 |
| `MaxPooling2D(2)` | (22, 22, 32) | 0 | |
| `Conv2D(64, 3)` + ReLU | (20, 20, 64) | 18 496 | (3 × 3 × 32 + 1) × 64 |
| `MaxPooling2D(2)` | (10, 10, 64) | 0 | |
| `Flatten` | (6400,) | 0 | 10 × 10 × 64 |
| `Dense(7)` + softmax | (7,) | 44 807 | 6 400 × 7 + 7 |
| **Total** | | **64 135** | |

**Pourquoi cette architecture ?**
- C'est la structure proposée par l'énoncé : un point de départ simple, que l'on peut expliquer couche par couche.
- Deux blocs suffisent pour des images de 48 × 48 : après deux *poolings*, les cartes ne font plus que 10 × 10.
- On double le nombre de filtres (32 → 64) quand la taille des cartes diminue : les motifs deviennent plus variés à mesure qu'ils sont plus grands.
- Le CNN a environ **4,6 fois moins de paramètres** que la baseline dense (64 135 contre 295 943), tout en tenant compte du voisinage des pixels.
- Les variantes (plus de filtres, couche `Dense` cachée, *dropout*…) seront testées une par une en Partie 6.

## Partie 4 — Entraînement du CNN

### Les choix d'entraînement et leur justification

| Choix | Valeur | Pourquoi |
|---|---|---|
| Fonction de perte | `categorical_crossentropy` | Chaque image a une seule classe parmi 7, codée en one-hot (étape 1c), et la sortie est un softmax. La perte vaut $-\log(p_{\text{vraie classe}})$ : elle est petite quand le modèle donne une forte probabilité à la bonne classe et devient très grande quand il se trompe avec assurance. |
| Optimiseur | Adam, taux d'apprentissage 0,001 (valeur par défaut de Keras) | Adam est une descente de gradient qui adapte le pas de chaque poids. Il fonctionne bien sans réglage sur un petit réseau comme le nôtre. On garde le taux par défaut car la perte de validation baisse dès les premières époques (courbes 4b) : rien n'indique un pas trop grand ou trop petit. En Partie 6, on teste de le faire baisser en cours d'entraînement (E0). |
| Taille de lot (*batch size*) | 128 | Une mise à jour des poids tous les 128 exemples, soit 180 mises à jour par époque (22 958 / 128). Avec des lots plus petits, chaque époque est plus longue et le gradient plus bruité ; avec des lots plus grands, on fait moins de mises à jour par époque. Sur GPU, 128 va plus vite que 64, et 128 images de 48 × 48 tiennent sans problème en mémoire. |
| Nombre d'époques | 30 au maximum, arrêt automatique (patience 6) | On ne connaît pas le bon nombre d'époques à l'avance. On met un plafond large et c'est `val_loss` qui décide quand s'arrêter (détails en 4a). Patience de 6 plutôt que 3 : la `val_loss` bouge un peu d'une époque à l'autre, et avec 3 on risque de s'arrêter sur un simple creux alors que le modèle peut encore progresser (on le soupçonnait pour E4). En Partie 6, le learning rate baisse quand la `val_loss` stagne : il faut aussi laisser quelques époques au modèle pour en profiter. |
| Métriques | accuracy pendant l'entraînement, puis rappel par classe en Partie 5 | L'accuracy est simple à lire et se compare directement à la baseline. Elle ne suffit pas avec des classes déséquilibrées : un modèle peut avoir une accuracy correcte en ne reconnaissant presque jamais `disgust`. C'est pour cela qu'on regarde aussi la matrice de confusion et le rappel de chaque classe. |

On surveille `val_loss` plutôt que `val_accuracy` pour l'arrêt automatique : la perte tient compte de la confiance du modèle et bouge plus finement d'une époque à l'autre, alors que l'accuracy ne change que quand une prédiction bascule.

### 4a — Entraîner le CNN avec arrêt automatique

On entraîne le CNN comme la baseline, avec une différence : l'**arrêt automatique** (*early stopping*). À la fin de chaque époque, Keras regarde `val_loss`. Si elle ne s'est pas améliorée depuis 6 époques (`patience=6`), l'entraînement s'arrête et on reprend les poids de la **meilleure époque** (`restore_best_weights=True`).

**Pourquoi ?** Passé un certain point, le modèle apprend par cœur les images d'entraînement : `loss` continue de baisser mais `val_loss` remonte. Le modèle qu'on veut garder est celui d'avant ce point. `EPOCHS_CNN` n'est donc qu'un maximum.

### 4b — Courbes d'apprentissage du CNN

On trace les mêmes courbes que pour la baseline (2c). La ligne en pointillés marque la meilleure époque, celle dont l'arrêt automatique a gardé les poids. Le graphique est mis dans une fonction `tracer_courbes`, réutilisée en Partie 6.

**Lecture des courbes.** Pendant les premières époques, les deux pertes baissent ensemble : ce que le modèle apprend sert aussi sur des images qu'il n'a jamais vues. Ensuite elles se séparent. La perte d'entraînement continue de descendre, alors que la perte de validation ne baisse presque plus, atteint son minimum (ligne en pointillés) puis remonte. Pour l'accuracy, l'entraînement continue de monter pendant que la validation plafonne autour de 50 %.

C'est du surapprentissage : après la meilleure époque, le modèle ne progresse plus que sur les images d'entraînement, qu'il commence à apprendre par cœur. L'arrêt automatique stoppe l'entraînement 6 époques après ce minimum et remet les poids de la meilleure époque. La cellule précédente affiche la meilleure époque et l'écart entre entraînement et validation pour cette exécution. Réduire cet écart est l'objet de plusieurs expériences de la Partie 6 (*dropout*, augmentation de données).

## Partie 5 — Évaluation du CNN

Toute cette partie utilise la **validation**. Le jeu de test reste réservé à l'évaluation finale du modèle retenu, une seule fois.

### 5a — Matrice de confusion et rappel par classe

L'accuracy donne un seul chiffre. Comme les classes sont déséquilibrées, un modèle peut avoir une accuracy correcte en ratant presque toujours `disgust`. La **matrice de confusion** montre le détail :
- chaque **ligne** est une vraie classe, chaque **colonne** une classe prédite ;
- la **diagonale** compte les bonnes réponses ; toutes les autres cases sont des erreurs, et on voit **avec quelle classe** le modèle confond.

Le **rappel** (*recall*) d'une classe est la part de ses images que le modèle a retrouvées : case de la diagonale ÷ total de la ligne.

### 5b — Courbes ROC (une classe contre toutes les autres)

Une courbe ROC se trace pour une question à deux réponses. Avec 7 classes, on en trace donc 7 : pour chacune, « l'image est-elle de cette classe, **oui ou non** ? » (*one-vs-rest*).

Pour une classe, par exemple `happy`, on décide « oui » quand la probabilité prédite dépasse un **seuil**. En faisant varier ce seuil de 1 à 0, on obtient pour chaque valeur :
- le **taux de vrais positifs** (axe vertical) : part des vraies images `happy` reconnues — c'est le rappel ;
- le **taux de faux positifs** (axe horizontal) : part des autres images prises à tort pour `happy`.

Plus la courbe monte vite vers le coin en haut à gauche, mieux la classe est séparée des autres. L'**AUC** (aire sous la courbe) résume la courbe en un nombre : 1,0 = séparation parfaite, 0,5 = hasard (la diagonale en pointillés).

**Attention :** l'AUC mesure si le modèle donne des probabilités plus élevées aux bonnes images, pas s'il prédit finalement la bonne classe. Une classe rare comme `disgust` peut avoir une AUC élevée et un rappel faible : il faut lire les courbes ROC **avec** la matrice de confusion.

### 5c — Classes les plus confondues et classes les plus difficiles

On lit la matrice de confusion hors de la diagonale. Le code liste toutes les cases d'erreur (vraie classe → classe prédite) et les trie de la plus grande à la plus petite. Chaque nombre est aussi donné en pourcentage de sa vraie classe, car les classes n'ont pas la même taille : 50 erreurs pèsent beaucoup plus pour `disgust` (73 images) que pour `happy`.

On classe ensuite les 7 classes de la plus difficile à la plus facile, selon leur rappel.

**Ce que montrent la matrice et ces chiffres.** Les valeurs exactes sont affichées par la cellule précédente ; elles varient un peu d'une exécution à l'autre, mais les tendances ci-dessous se retrouvent à chaque fois.

`happy` et `surprise` sont les classes les mieux reconnues. Leurs signes sont très visibles, même en 48 × 48 : la bouche qui sourit pour l'une, la bouche ouverte et les yeux écarquillés pour l'autre. `happy` est aussi la classe qui a le plus d'images d'entraînement.

Les confusions les plus fréquentes font presque toutes intervenir `sad` et `neutral` : `neutral → sad`, `fear → sad`, `sad → neutral`, `angry → sad`. La colonne `sad` de la matrice reçoit beaucoup d'images des autres classes. Un visage triste, neutre ou inquiet a souvent la bouche fermée et peu de mouvement : la différence tient à quelques pixels autour des sourcils et des coins de la bouche.

Les classes les plus difficiles :
- `disgust`. C'est d'abord un problème de quantité : 436 images d'entraînement contre 7 215 pour `happy`. Le modèle voit rarement cette classe et se tromper sur elle coûte peu à l'accuracy globale. Quand il se trompe, il répond souvent `angry` : dans les deux expressions, les sourcils se froncent et le nez se plisse.
- `fear`. Elle n'a pas de signe qui lui soit propre : les yeux grands ouverts et la bouche ouverte la font prendre pour `surprise`, une bouche crispée pour `sad`. Ses erreurs sont réparties sur toutes les autres classes.
- `angry`, souvent prise pour `sad` ou `neutral`, probablement parce que des sourcils froncés se voient mal à cette résolution.

Une autre cause probable, que nous n'avons pas mesurée, est le bruit dans les étiquettes. Les images de FER2013 viennent de recherches sur le web et certaines expressions sont ambiguës même pour un humain : une partie des « erreurs » du modèle n'en sont peut-être pas (voir 5d).

### 5d — Exemples : image, vraie classe, classe prédite, probabilité

On affiche 4 images bien classées (en vert) et 8 erreurs (en rouge). Pour les erreurs, on prend celles où le CNN était **le plus sûr de lui** : ce sont les plus instructives, puisque le modèle s'y trompe avec une forte probabilité. Sous chaque image : la vraie classe avec la probabilité que le modèle lui a donnée, puis la classe prédite avec sa probabilité.

**Analyse des erreurs.** Pour les erreurs affichées, le modèle donne presque toute la probabilité à la mauvaise classe et presque rien à la vraie. Les images exactes changent un peu d'une exécution à l'autre, mais on retrouve toujours deux sortes d'erreurs.

Certaines viennent d'expressions qui se ressemblent. Des images `fear` sont prédites `surprise` : yeux écarquillés et bouche grande ouverte, parfois avec les mains sur les joues, c'est-à-dire exactement ce que le modèle associe à la surprise. Une bouche qui crie, dents visibles, fait prendre `fear` pour `angry`. Une tête appuyée sur la main, les yeux baissés, fait prendre `neutral` pour `sad`.

D'autres semblent venir d'étiquettes fausses. Des images étiquetées `surprise`, `sad` ou `neutral` montrent des personnes qui sourient ou rient et sont prédites `happy` : à l'œil, la prédiction paraît plus juste que l'étiquette. Une grimace qui montre les dents peut aussi être confondue avec un sourire.

Le modèle semble donc s'appuyer surtout sur la forme de la bouche et l'ouverture des yeux, et moins sur des indices plus fins comme la position des sourcils. Les étiquettes douteuses, elles, limitent probablement le score que n'importe quel modèle peut atteindre sur FER2013.

## Partie 6 — Expériences

Le CNN de la Partie 4 (`cnn_base`) sert de référence. Chaque expérience part du **dernier modèle retenu** et ne change qu'une chose, pour pouvoir attribuer l'effet observé à ce changement. Tout le reste est identique :

- même découpage train / validation, même graine (remise à 42 avant de créer chaque modèle) ;
- même compilation (Adam, taux 0,001, `categorical_crossentropy`), mêmes lots de 128 ;
- 20 époques fixes pour chaque expérience, en gardant les poids de la meilleure époque (plus petite `val_loss`) ; seul `cnn_base` garde l'arrêt automatique de la Partie 4 ;
- E0 teste un learning rate qui baisse pendant l'entraînement (`ReduceLROnPlateau`). Comme chaque expérience part de la précédente, toutes les expériences suivantes gardent ce réglage (`baisse_lr=True`) : sinon on comparerait des modèles entraînés de deux façons différentes ;
- tous les scores sont mesurés sur la **validation**. Le jeu de test n'est utilisé qu'une fois, à la fin, pour le modèle choisi.

Pour chaque modèle on note l'accuracy et la perte de validation, la meilleure époque, l'écart d'accuracy entre entraînement et validation à cette époque (signe de surapprentissage) et le rappel de `disgust`, la classe la plus rare.

| Expérience | Changement | Hypothèse |
|---|---|---|
| E0 | un learning rate qui baisse pendant l'entraînement (`ReduceLROnPlateau`) | des pas plus petits quand la perte de validation ne baisse plus devraient la faire descendre un peu plus bas |
| E1 | deux couches `Dense(128)` et `Dense(64)` + ReLU avant la sortie (la figure 2 de l'énoncé en a une) | des couches cachées combinent les motifs trouvés par les convolutions avant de décider |
| E1 bis | une `BatchNormalization` après chaque convolution | des valeurs recentrées entre les couches devraient rendre l'entraînement plus régulier |
| E2 | un `Dropout(0.3)` juste avant cette couche `Dense(128)` | le modèle surapprend dès les premières époques (4b) ; le *dropout* devrait réduire l'écart entre entraînement et validation |
| E3 | un troisième et un quatrième bloc, `Conv2D(128)` puis `Conv2D(256)`, + `MaxPooling` | des motifs plus grands, et beaucoup moins de valeurs à l'entrée de la couche dense |
| E4 | de l'augmentation de données (miroir horizontal, petite rotation, petit zoom, flou léger) | le modèle voit des variantes des visages à chaque époque et apprend moins par cœur |
| E5 | E4 + une pondération des classes (`class_weight`) | une erreur sur `disgust` coûte plus cher, son rappel devrait monter, sans doute au prix d'un peu d'accuracy |

Les critères de choix sont fixés avant de regarder les résultats : on garde un changement s'il améliore l'accuracy de validation ; à accuracy presque égale (moins d'un point d'écart), on regarde la perte de validation et l'écart entre entraînement et validation. Le modèle final est celui qui a la meilleure accuracy de validation : c'est le code qui le choisit en 6h, pour que le choix ne dépende pas de ce qu'on a envie de voir.

### 6a — Deux petites fonctions pour ne pas répéter le code

- `entrainer(modele, poids_classes=None, baisse_lr=False, epoques=20)` compile et entraîne un modèle avec les mêmes réglages que le CNN de la Partie 4, mais sur un **nombre d'époques fixe** (20 par défaut) : tous les modèles ont ainsi le même temps d'entraînement et aucun n'est arrêté trop tôt. À la fin, on reprend quand même les poids de la meilleure époque (plus petite `val_loss`). Avec `epoques=None`, on retrouve l'arrêt automatique de la Partie 4. Avec `baisse_lr=True`, il ajoute la baisse du learning rate testée en E0.
- `mesurer(nom, changement, modele, historique)` calcule les scores de validation et ajoute une ligne à la liste `resultats`, qui deviendra le tableau comparatif.

On mesure tout de suite `cnn_base`, déjà entraîné en 4a : c'est la première ligne du tableau.

### 6a bis — Expérience 0 : un learning rate qui baisse (`ReduceLROnPlateau`)

**Hypothèse.** Avec `cnn_base`, le learning rate d'Adam reste à 0,001 du début à la fin. Au début, de grands pas font baisser la perte vite. Près du minimum, ces mêmes pas sont trop grands : le modèle saute autour du minimum sans s'y poser et la `val_loss` cesse de baisser. Avec des pas plus petits à ce moment-là, la `val_loss` devrait descendre un peu plus bas.

**Changement.** Même modèle que `cnn_base`, entraîné avec le callback `ReduceLROnPlateau` : quand la `val_loss` n'a pas baissé depuis 3 époques (`patience=3`), le learning rate est divisé par 2 (`factor=0.5`), sans descendre sous 0,00001 (`min_lr`). On avait d'abord mis `patience=1` : la `val_loss` remonte souvent un peu sur une seule époque sans que ce soit un vrai plateau, donc le learning rate était divisé trop souvent, devenait presque nul, et l'entraînement s'arrêtait trop tôt. On ne touche pas à l'architecture : c'est un changement dans l'entraînement seulement. L'entraînement dure 20 époques fixes, comme pour toutes les expériences.

### 6b — Expérience 1 : deux couches cachées `Dense(128)` et `Dense(64)`

**Hypothèse.** Dans `cnn_base`, les 6 400 valeurs sorties des convolutions vont directement vers les 7 sorties. Des couches cachées peuvent d'abord combiner ces motifs (par exemple des coins de bouche relevés avec des yeux plissés) avant de décider. La figure 2 de l'énoncé en a une ; on en met deux, pour descendre par étapes de 6 400 valeurs vers 7 au lieu de tout résumer d'un coup.

**Changement.** `Flatten → Dense(128, relu) → Dense(64, relu) → Dense(7, softmax)` au lieu de `Flatten → Dense(7, softmax)`. Le nombre de paramètres passe de 64 135 à 847 367, presque tous dans la première couche (6 400 × 128 + 128 = 819 328). La couche `Dense(64)` n'en ajoute que 8 256 (128 × 64 + 64).

### 6b bis — Expérience 1 bis : *batch normalization*

**Hypothèse.** À chaque mise à jour des premières couches, les couches suivantes reçoivent des valeurs qui changent de plage et doivent se réadapter. Si on recentre ces valeurs entre les couches, l'entraînement devrait être plus régulier et la perte de validation descendre plus bas. On le teste tôt dans la série, car cela comptera encore plus quand on ajoutera des blocs de convolution (E3).

**Changement.** On repart de E1. Après chaque convolution, on ajoute une couche `BatchNormalization` : elle recentre les valeurs de chaque carte sur le lot (moyenne 0, écart-type 1), puis apprend une échelle et un décalage. L'ordre dans un bloc devient convolution → *batch normalization* → ReLU → pooling. On ne l'ajoute qu'aux convolutions, pour ne changer qu'une chose.

À la prédiction, la couche n'utilise pas les statistiques du lot mais des moyennes mémorisées pendant l'entraînement : une image seule est donc traitée de la même façon qu'un lot.

Le modèle passe de 847 367 à 847 751 paramètres (4 par filtre, dont 2 appris). Les expériences suivantes (E2 à E5) repartent de ce modèle : elles ont donc toutes la *batch normalization*.

### 6c — Expérience 2 : *Dropout*

**Hypothèse.** Avec E1, la meilleure époque arrive tôt ; ensuite la perte de validation remonte alors que l'accuracy d'entraînement continue de grimper. Presque tous les paramètres sont dans la couche `Dense(128)` (819 328 sur 847 751). Le *dropout* devrait freiner l'apprentissage par cœur dans cette couche et repousser la meilleure époque.

**Changement.** Un `Dropout(0.3)` entre `Flatten` et `Dense(128)`. Pendant l'entraînement, il met à zéro au hasard environ 30 % des 6 400 valeurs, différemment à chaque lot : le réseau ne peut plus compter sur quelques valeurs précises et doit répartir l'information. À la prédiction, le *dropout* est désactivé et toutes les valeurs sont utilisées. Il n'ajoute aucun paramètre.

On l'a placé avant `Dense(128)` parce que c'est là que se trouvent presque tous les paramètres du modèle, donc l'essentiel du risque d'apprendre par cœur.

**Pourquoi 0,3 et pas 0,5 ?** 0,5 est la valeur la plus courante, mais elle retire la moitié de l'information à chaque lot, ce qui est beaucoup pour un petit modèle. On l'avait constaté dans notre première exécution, avec un *dropout* de 0,5 : une fois l'augmentation ajoutée (E4), le modèle peinait à apprendre, son accuracy d'entraînement passait sous celle de validation. Avec 0,3, on cherche à freiner le surapprentissage sans trop ralentir l'apprentissage.

### 6d — Expérience 3 : un troisième et un quatrième bloc de convolution

**Hypothèse.** Avec deux blocs, chaque valeur de la dernière carte ne « voit » qu'un morceau du visage. Deux blocs de plus combinent ces morceaux en motifs plus grands (la bouche avec les joues, les sourcils avec les yeux). Ils réduisent aussi la taille des cartes avant `Flatten` : la couche `Dense(128)` reçoit 1 024 valeurs au lieu de 6 400, donc le modèle a moins de poids à apprendre par cœur.

**Changement.** Un bloc `Conv2D(128, 3)` + `MaxPooling2D(2)`, puis un bloc `Conv2D(256, 3)` + `MaxPooling2D(2)`, après le deuxième bloc. On double le nombre de filtres à chaque bloc (32 → 64 → 128 → 256), comme en Partie 3. Chaque convolution est suivie de la *batch normalization*, comme depuis E1 bis. La `Conv2D(256)` utilise `padding="same"` pour garder la taille (4, 4) : sans cela il ne resterait qu'une seule valeur par carte après le pooling.

| Couche | Forme de sortie | Paramètres |
|---|---|---:|
| après le 2e bloc | (10, 10, 64) | |
| `Conv2D(128, 3)` + ReLU | (8, 8, 128) | (3 × 3 × 64 + 1) × 128 = 73 856 |
| `MaxPooling2D(2)` | (4, 4, 128) | 0 |
| `Conv2D(256, 3, padding="same")` + ReLU | (4, 4, 256) | (3 × 3 × 128 + 1) × 256 = 295 168 |
| `MaxPooling2D(2)` | (2, 2, 256) | 0 |
| `Flatten` | (1024,) | 0 |
| `Dense(128)` | (128,) | 1 024 × 128 + 128 = 131 200 |
| `Dense(64)` | (64,) | 128 × 64 + 64 = 8 256 |
| `BatchNormalization` après les 4 convolutions | | 4 × (32 + 64 + 128 + 256) = 1 920 |
| **Total du modèle** | | **530 183** (contre 847 751 pour E2) |

### 6e — Expérience 4 : augmentation de données

**Hypothèse.** E3 surapprend encore : à sa meilleure époque, l'accuracy d'entraînement dépasse celle de validation. Avec l'augmentation, le modèle ne voit jamais deux fois exactement la même image, ce qui devrait réduire cet écart et, peut-être, améliorer la validation.

**Changement.** Quatre couches Keras au début du modèle, qui transforment chaque image au hasard, différemment à chaque époque :
- `RandomFlip("horizontal")` : miroir gauche-droite. Une expression reste la même dans un miroir. On ne retourne pas haut-bas, car il n'y a pas de visages à l'envers dans les données ;
- `RandomRotation(0.05)` : rotation d'au plus 0,05 tour, soit ± 18°, comme une tête un peu penchée ;
- `RandomZoom(0.1)` : zoom avant ou arrière d'au plus 10 %, comme un visage recadré un peu plus large ou plus serré ;
- `RandomGaussianBlur(factor=0.5, kernel_size=3, sigma=1.0, value_range=(0, 1))` : flou gaussien léger (noyau 3 × 3), plus ou moins fort selon l'image. Les photos de FER2013 n'ont pas toutes la même netteté : le modèle doit reconnaître l'expression même sur une image un peu floue. `value_range=(0, 1)` indique que nos pixels sont déjà divisés par 255.

Ces couches ne sont **actives que pendant l'entraînement** : en validation, en test et pour la démonstration, l'image passe sans modification. Le nombre d'images par époque ne change pas, ce sont leurs variantes qui changent d'une époque à l'autre.

### 6f — Expérience 5 : pondération des classes

**Hypothèse.** Même avec E4, `disgust` reste une classe mal reconnue. Le modèle voit cette classe 16 fois moins souvent que `happy`, donc ses erreurs sur `disgust` pèsent peu dans la perte. En donnant plus de poids aux images des classes rares, son rappel devrait monter, sans doute au prix d'un peu d'accuracy sur les classes fréquentes.

**Changement.** Même modèle que E4, le dernier retenu (avec l'augmentation), entraîné en plus avec `class_weight`. On compare E5 à E4 : la seule différence est la pondération, donc l'effet observé vient d'elle. Le poids d'une classe vaut $\dfrac{\text{nombre total d'images}}{7 \times \text{nombre d'images de la classe}}$ : une classe de taille moyenne a un poids proche de 1, une classe rare un poids plus grand. Dans la perte, une erreur sur une image est multipliée par le poids de sa classe.

On calcule ces poids avec les comptes de l'étape 1a (`nombres_train`, dossier `train` complet). La validation est tirée au hasard dans ce dossier, donc les proportions sont presque les mêmes que dans la partie entraînement.

### 6g — Tableau comparatif (validation)

Le tableau est affiché à partir de la liste `resultats`. « Écart » = accuracy d'entraînement moins accuracy de validation, à la meilleure époque. Avec *dropout* ou augmentation, l'accuracy d'entraînement est mesurée sur des images perturbées : l'écart y paraît donc un peu plus petit qu'il ne l'est.

Sous le tableau, la cellule affiche aussi la **précision** et le **F1** de chaque classe, pour chaque modèle. Le rappel dit quelle part des images d'une classe le modèle retrouve (la ligne de la matrice de confusion). La précision regarde l'autre sens : parmi les images que le modèle a prédites dans cette classe, quelle part en est vraiment (la colonne). Un modèle qui répond très souvent `disgust` aura un bon rappel sur `disgust` mais une mauvaise précision. Le F1 résume les deux en un seul nombre, $F_1 = \dfrac{2 \times \text{précision} \times \text{rappel}}{\text{précision} + \text{rappel}}$ : il n'est élevé que si les deux le sont. Si un modèle ne prédit jamais une classe, sa précision n'est pas définie et on affiche 0.

### 6h — Choix et courbes du modèle final

Le code applique la règle fixée au début de la Partie 6 : il parcourt `resultats` et garde le modèle qui a la meilleure accuracy de validation. Si un autre modèle retrouve beaucoup mieux `disgust` en perdant un peu d'accuracy, on le présente comme l'alternative « si le but était de détecter le dégoût », sans changer la règle après coup. On retrace ensuite les courbes du modèle choisi pour les comparer à celles du CNN de base (4b).

### 6i — Évaluation finale sur le jeu de test

**Attention : c'est la seule fois du projet où le jeu de test est utilisé.** Toutes les décisions (architecture, *dropout*, augmentation, pondération, choix du modèle final) ont été prises avant, sur la validation. Si on se servait du test pour choisir entre plusieurs modèles, son score ne serait plus une estimation honnête de ce que le modèle fait sur des images jamais vues. Cette cellule ne doit donc pas servir à revenir modifier le modèle.

On mesure l'accuracy, la matrice de confusion et le rappel par classe, puis on sauvegarde le modèle dans `models/modele_final.keras` (dossier non versionné) pour la démonstration.

### Démonstration : prédire l'expression d'une image

Cette cellule recharge le modèle sauvegardé et prédit l'expression d'une seule image. Pour essayer une autre image, il suffit de changer `CHEMIN_IMAGE`.

L'image est préparée **exactement comme pendant l'entraînement** : niveaux de gris, 48 × 48 pixels, valeurs divisées par 255. Sans la division par 255, le modèle recevrait des valeurs 255 fois trop grandes et ses prédictions n'auraient plus de sens. Le modèle a appris sur des visages déjà recadrés : sur une photo où le visage est petit ou décentré, il se trompera (c'est le rôle de la détection de visages des parties 8 et 9).

---

# 2. Questions du jury sur la Partie 1

Pour chaque question : la réponse à donner, puis le pourquoi (chaîne de cause à effet), puis le piège ou la relance probable.

## Q1. Pourquoi diviser les pixels par 255 et encoder les labels en one-hot ?

**Réponse.** On divise par 255 pour ramener les pixels entre 0 et 1, l'échelle pour laquelle l'initialisation des poids et le learning rate d'Adam sont prévus. Les labels sont en one-hot pour avoir 7 nombres, comme les 7 sorties du softmax.

**Pourquoi.**
1. Un neurone calcule z = w1·x1 + … + w2304·x2304 + b. Si tous les x sont multipliés par 255, z l'est aussi.
2. Au départ les biais valent 0 et ReLU(255·a) = 255·ReLU(a) : sans /255, chaque activation cachée et chaque score final est 255 fois plus grand.
3. Softmax sature : [2, 1, 0] donne environ [0,67 ; 0,24 ; 0,09], mais [200, 100, 0] donne environ [1 ; 0 ; 0]. Le réseau met presque toute la probabilité sur une classe prise au hasard dès le départ, et la perte initiale est énorme.
4. Adam déplace chaque poids d'environ 0,001 par mise à jour : avec des pixels bruts, ce pas fait bouger z 255 fois plus, l'apprentissage devient instable.
5. `categorical_crossentropy` vaut −Σ yk·log(pk) ; avec un one-hot il ne reste que −log(p de la vraie classe) (p = 0,6 donne 0,51, p = 0,01 donne 4,6).
6. Diviser par 255 ne retire aucune information : c'est un changement d'unité, le même pour toutes les images.

**Relances.** « Avec des labels entiers, ça ne marcherait pas ? » Si : `label_mode="int"` avec `sparse_categorical_crossentropy`, c'est le même calcul. « Pourquoi 255 et pas la moyenne et l'écart-type ? » 255 est une constante qui ne dépend d'aucune donnée, donc aucun risque de fuite ; une standardisation devrait être calculée sur le train seulement.

## Q2. Que fait le filtre des images vides ?

**Réponse.** Il supprime les images presque d'une seule couleur (écart-type des pixels inférieur à 1) : 13 images sans visage, dont l'étiquette est forcément fausse.

**Pourquoi.**
1. L'écart-type mesure de combien les pixels s'écartent typiquement de leur moyenne, calculé sur les pixels bruts (0 à 255). Une image unie vaut environ 0 ; un visage (peau claire, yeux et cheveux sombres) bien plus.
2. L'écart-type attrape toutes les images unies, noires, blanches ou grises ; une moyenne nulle n'attraperait que les noires.
3. Le seuil est 1 et pas 0 parce que la compression JPEG ajoute un bruit minuscule.
4. Une image unie ne contient aucune forme : elle ne peut rien apprendre sur un visage. 13 images sur 35 887, l'effet sur le score est négligeable : la vraie raison est la qualité des étiquettes.
5. Un seuil trop haut (30) risquerait de supprimer de vrais visages sombres ou peu contrastés (non mesuré).

**Piège.** `unlink()` efface le fichier du disque, ce n'est pas un filtre en mémoire. Si on relance, la cellule affiche « images vides retirées: 0 ». Pour tout récupérer : supprimer `data/fer2013` et relancer l'étape 0.

## Q3. `label_mode="categorical"` ou `"int"`, et à quoi sert `class_names` ?

**Réponse.** `"categorical"` donne des labels de forme (128, 7), comme la sortie du modèle ; `"int"` donne (128,). `class_names` fixe que le neurone 3 est `happy` partout.

**Pourquoi.**
1. Avec `"int"` et `categorical_crossentropy`, `fit` plante dès l'entraînement de la baseline (formes différentes).
2. Si on passe en `sparse_categorical_crossentropy`, ça tourne, mais `labels_onehot.argmax(axis=1)` plante en 5a et le repère de 2c afficherait « toujours angry : 14,3 % » au lieu de happy, sans message d'erreur.
3. Sans `class_names`, Keras range les dossiers par ordre alphabétique, qui est déjà celui de `CLASS_NAMES` : rien ne change ici. Le paramètre rend l'ordre explicite, le partage avec les poids de classes (6f) et la démo, et provoque une erreur si un nom ne correspond à aucun dossier.

**Piège.** Répondre « sans `class_names` les classes seraient mélangées » est faux.

## Q4. Rôle de train, validation et test, et conséquences du déséquilibre

**Réponse.** Le train sert à apprendre les poids, la validation à suivre l'entraînement et à choisir entre les modèles, le test à juger une seule fois le modèle final. Le déséquilibre pousse le modèle vers `happy` et rend l'accuracy trompeuse.

**Pourquoi.**
1. Choisir un modèle sur un jeu de données revient un peu à apprendre dessus : il faut un test que personne n'a regardé.
2. La perte d'un lot est une moyenne. Un lot contient en moyenne environ 16 fois plus d'images `happy` que `disgust` (7 215 / 436 ≈ 16,5) : les images `happy` orientent plus souvent les mises à jour, et la sortie penche vers `happy`.
3. `disgust` fait environ 70 images sur 5 739 en validation : ne jamais la trouver ne coûte qu'environ 1,2 point d'accuracy. D'où le rappel par classe (bonnes réponses / images de la classe) en 5a.
4. `class_weight` (E5) multiplie la perte de chaque image par le poids de sa classe : 9,4 pour `disgust`, environ 0,57 pour `happy`.
5. La validation n'est pas stratifiée : environ 70 images `disgust` au lieu de 87 ; une image fait bouger son rappel d'environ 1,4 point.

**Relances.** « Pourquoi ne pas rééquilibrer aussi la validation et le test ? » Ils doivent refléter la vraie répartition, sinon le score ne mesure plus la performance réelle. « Et si vous dupliquiez les images `disgust` avant le découpage ? » Des copies se retrouveraient en train et en validation : fuite de données, rappel gonflé.

## Q5. Que font `validation_split`, `subset="both"` et `seed` ?

**Réponse.** Un seul mélange et une seule coupe : aucune image ne peut être à la fois en train et en validation, et la même seed redonne le même découpage.

**Pourquoi.**
1. Keras liste les 28 697 fichiers (après suppression des images vides), les mélange une fois avec la seed 42 et met les 5 739 derniers (int(0,2 × 28 697)) en validation, 22 958 en train.
2. Sans seed, Keras lève une erreur pour éviter que train et validation se chevauchent.
3. Avec deux appels séparés et deux seeds différentes, environ 80 % des images de validation seraient aussi en train : fuite.
4. `set_random_seed(SEED)` fixe les poids de départ, le dropout et l'augmentation ; la seed du chargeur fixe quelles images vont où et l'ordre de mélange du train.
5. 22 958 / 128 donne 180 lots par époque, le dernier n'a que 46 images.

## Q6. Le redimensionnement et le canal gris

**Réponse.** La seule couche dont la taille dépend de l'image est la Dense après le Flatten : toutes les images doivent donc avoir la forme (48, 48, 1). Le chargeur s'en charge.

**Pourquoi.**
1. `image_dataset_from_directory` redimensionne toujours à `image_size` (défaut 256 × 256). On lui donne la taille réelle, 48 × 48, donc rien n'est déformé.
2. Une Dense a un poids par valeur d'entrée et par neurone : 2 304 × 128 pour la baseline, 6 400 × 7 pour le CNN de base.
3. En 256 × 256, la baseline aurait 65 536 × 128 = 8 388 608 poids dans sa première couche, et `Input(shape=(48, 48, 1))` refuserait ces images.
4. Le nombre de poids d'une convolution ne dépend pas de la taille d'image ((3×3×1+1)×32 = 320) ; la taille des feature maps, elle, en dépend.
5. `color_mode="grayscale"` donne 1 canal ; le défaut `rgb` donnerait (48, 48, 3), trois copies du même gris, refusées par le modèle.

**Piège.** Une image en 64 × 64 serait redimensionnée sans prévenir : le Flatten ne planterait pas. La vérification de 1b prouve qu'aucune image n'est déformée sans qu'on le sache.

## Q7. `cache`, `prefetch` et `shuffle=False`

**Réponse.** Le train doit être remélangé à chaque époque ; la validation et le test ont un ordre fixe dès leur création ; le cache et le prefetch ne servent qu'à aller plus vite.

**Pourquoi.**
1. Keras mélange la liste des fichiers une fois, puis remélange localement le train à chaque époque (tampon de 8 lots).
2. `.cache()` rejoue exactement ce qui est sorti au premier passage : sur le train, il figerait l'ordre et on perdrait le mélange.
3. Avec `subset="both"`, Keras crée la validation avec `shuffle=False` ; le test a `shuffle=False` dans notre code. Leur ordre est fixe même sans cache.
4. Le cache garde en mémoire les images déjà divisées par 255 : on ne relit plus les fichiers JPEG.
5. Ordre fixe = labels et prédictions alignés même en deux passages (6i). En 5a, images, labels et prédictions sont récupérés dans la même boucle, donc alignés quoi qu'il arrive.

## Q8. Avez-vous « touché au test » ? Où une fuite pourrait-elle se glisser ?

**Réponse.** Retirer l'image vide du test n'est pas une fuite : on n'a lu que des pixels, avec une règle générique fixée avant tout entraînement, qui ne regarde ni label ni score, appliquée pareil partout. On le déclare : 7 177 images au lieu de 7 178.

**Pourquoi.**
1. Les deux vraies fuites possibles : deux découpages avec deux seeds différentes, et choisir le modèle d'après le test.
2. Exemple d'honnêteté : les poids de classes (6f) sont calculés sur tout le dossier `train`, validation comprise ; seules les proportions des classes passent, pas les images.
3. La validation a servi à l'arrêt automatique et au choix entre plusieurs modèles : son score est un peu optimiste. C'est pour ça que le test ne sert qu'une fois, en 6i.
4. Un écart d'environ un demi-point entre test et validation est du même ordre que l'erreur de mesure sur 7 177 images (environ 0,6 point) : on peut dire que le test confirme la validation, pas qu'elle était optimiste.

**Relance (étudiant solide).** « Avez-vous vérifié les doublons ou la même personne entre train et test ? » Non : les images viennent du web, le score de test peut donc être un peu optimiste.
