# ASL Sign Language Detector

Détection en temps réel de l'alphabet en langue des signes américaine (ASL) via webcam, utilisant un réseau de neurones convolutif (CNN) entraîné sur 87 000 images.

---

## Description
Ce projet, réalisé dans le cadre du module **Traitement d’Images** à l’**École Nationale des Sciences Appliquées de Tanger (ENSA Tanger)** sous l’encadrement de **M. Lachkar Abdelmounaim**, permet de :

- Reconnaître les 26 lettres de l'alphabet ASL (A–Z) + `space`, `del`, `nothing`
- Construire des mots lettre par lettre devant la caméra
- Entendre chaque lettre et le mot complet grâce à la synthèse vocale (pyttsx3)

**Stack technique :** Python · TensorFlow/Keras · OpenCV · scikit-learn · pyttsx3

---

## Structure du projet

```
asl_sign_language/
├── data/                        ← Déposer le dataset ici
├── src/
│   ├── preprocessing.py         ← Chargement + split + visualisation du dataset
│   ├── augmentation.py          ← Augmentation de données (rotations, zoom, luminosité)
│   ├── model.py                 ← Architecture CNN (3 blocs Conv + Dense)
│   ├── train.py                 ← Script principal d'entraînement
│   ├── evaluate.py              ← Courbes, matrice de confusion, rapport de classification
│   └── app.py                   ← Application webcam temps réel
├── models/                      ← Modèle sauvegardé automatiquement ici
├── outputs/                     ← Graphiques générés automatiquement ici
├── config.py                    ← Tous les hyperparamètres centralisés
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Cloner ou télécharger le projet

```bash
git clone <url-du-repo>
cd asl_sign_language
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Note GPU** : Pour entraîner sur GPU, installez `tensorflow-gpu` à la place de `tensorflow` et assurez-vous d'avoir les drivers CUDA/cuDNN compatibles.

---

## Dataset

### Où télécharger

1. Aller sur [Kaggle — American Sign Language Dataset (madhavanair)](https://www.kaggle.com/datasets/madhavanair/american-sign-language-dataset)
2. Télécharger et décompresser l'archive

### Où placer les fichiers

```
asl_sign_language/
└── data/
    └── asl_alphabet_train/       ← coller ce dossier ici
        ├── A/
        │   ├── A1.jpg
        │   ├── A2.jpg
        │   └── ...
        ├── B/
        ├── C/
        ├── ...
        ├── del/
        ├── nothing/
        └── space/
```

Le chemin est configurable dans [config.py](config.py) via `DATASET_PATH`.

---

## Utilisation

### Entraînement

```bash
python src/train.py
```

Ce script va :
1. Charger le dataset depuis `data/asl_alphabet_train/`
2. Afficher la distribution des classes et des exemples
3. Diviser en train (80%) / validation (10%) / test (10%)
4. Construire et entraîner le CNN avec augmentation de données
5. Sauvegarder le meilleur modèle dans `models/best_asl_model.h5`
6. Générer tous les graphiques d'évaluation dans `outputs/`

Durée estimée : ~30 min sur CPU, ~5 min sur GPU.

### Application webcam

```bash
python src/app.py
```

**Contrôles :**
| Touche | Action |
|--------|--------|
| `Q` | Quitter l'application |
| `S` | Dire le mot complet à voix haute |
| `C` | Effacer le mot en cours |

**Comment utiliser :**
1. Placez votre main dans le rectangle vert à l'écran
2. Maintenez le signe ASL stable pendant ~1.5 secondes
3. La lettre s'ajoute automatiquement au mot affiché en bas
4. Utilisez le signe `space` pour ajouter un espace, `del` pour effacer

---

## Description des fichiers

| Fichier | Rôle |
|---------|------|
| `config.py` | Centralise tous les hyperparamètres (taille image, LR, seuils…) |
| `src/preprocessing.py` | Charge les images depuis le disque, les normalise en niveaux de gris, divise le dataset |
| `src/augmentation.py` | Génère des variantes artificielles (rotation, zoom, luminosité) pour améliorer la robustesse |
| `src/model.py` | Définit l'architecture CNN : 3 blocs Conv2D + BatchNorm + Dropout + 2 couches Dense |
| `src/train.py` | Orchestre le pipeline complet : données → modèle → entraînement → évaluation |
| `src/evaluate.py` | Produit les courbes d'apprentissage, la matrice de confusion, le rapport de classification et les erreurs |
| `src/app.py` | Application temps réel : webcam → ROI → prédiction → mot → synthèse vocale |

---

## Pipeline complet

```
Dataset Kaggle (87 000 images, 29 classes)
         │
         ▼
  preprocessing.py
  ├─ Conversion BGR → Niveaux de gris
  ├─ Redimensionnement 64×64
  ├─ Normalisation [0,1]
  └─ Split stratifié 80/10/10
         │
         ▼
  augmentation.py
  ├─ Rotation ±15°
  ├─ Décalage ±10%
  ├─ Zoom ±10%
  └─ Luminosité ×[0.8,1.2]
         │
         ▼
     model.py (CNN)
  ├─ Bloc 1 : Conv(32) + BN + Pool + Dropout
  ├─ Bloc 2 : Conv(64) + BN + Pool + Dropout
  ├─ Bloc 3 : Conv(128) + BN + Pool + Dropout
  ├─ Dense(256) + BN + Dropout
  ├─ Dense(128) + Dropout
  └─ Dense(29, softmax)
         │
         ▼
     train.py
  ├─ EarlyStopping (patience=5)
  ├─ ModelCheckpoint (best val_acc)
  └─ ReduceLROnPlateau
         │
         ▼
    evaluate.py
  ├─ Courbes accuracy/loss
  ├─ Matrice de confusion
  ├─ Rapport de classification
  └─ Exemples d'erreurs
         │
         ▼
      app.py
  └─ Webcam → ROI → prédiction → mot → pyttsx3
```

---

## Hyperparamètres clés (config.py)

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `IMG_SIZE` | 64 | Compromis détail/vitesse |
| `BATCH_SIZE` | 32 | Standard, bon équilibre gradient/RAM |
| `EPOCHS` | 50 | EarlyStopping arrête avant si nécessaire |
| `LEARNING_RATE` | 0.001 | Défaut Adam (Kingma & Ba 2014) |
| `CONFIDENCE_THRESHOLD` | 0.85 | Évite les prédictions incertaines |
| `LETTER_DELAY` | 1.5s | Évite la répétition involontaire |

---

## Sorties générées

Après entraînement, le dossier `outputs/` contiendra :

| Fichier | Description |
|---------|-------------|
| `sample_images.png` | Grille d'exemples du dataset |
| `augmentation_examples.png` | Comparaison original vs augmenté |
| `training_curves.png` | Accuracy et loss train/val par époque |
| `confusion_matrix.png` | Matrice de confusion normalisée 29×29 |
| `classification_report.txt` | Précision, rappel, F1 par classe |
| `prediction_errors.png` | Exemples d'erreurs du modèle |
| `training_history.json` | Historique complet de l'entraînement |
