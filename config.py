# =============================================================================
# config.py — Paramètres centralisés du projet ASL
# =============================================================================
# Pourquoi centraliser ? Évite de disperser des "magic numbers" dans tout le
# code. Un seul endroit à modifier pour changer le comportement global.

import os

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

# Chemin vers le dossier contenant les sous-dossiers de classes
# L'utilisateur doit y déposer le dataset Kaggle décompressé
DATASET_PATH = "data/asl_alphabet_train/"

# Modèle Keras sauvegardé (meilleurs poids selon val_accuracy)
MODEL_PATH = "models/best_asl_model.h5"

# Dossiers de sortie (créés automatiquement si absents)
OUTPUTS_DIR = "outputs/"
MODELS_DIR  = "models/"

# ---------------------------------------------------------------------------
# Prétraitement & architecture
# ---------------------------------------------------------------------------

# Taille cible des images après center-crop et resize.
# 128×128 préserve bien plus de détail que 64×64 sur des images 480×640.
IMG_SIZE = 128

# Nombre de canaux : 3 = couleur BGR (plus discriminant que grayscale)
IMG_CHANNELS = 3

# Nombre d'images par mini-batch
# 32 : avec Transfer Learning la base est gelée, RAM suffisante pour 128×128×3
BATCH_SIZE = 32

# Nombre maximum d'images par classe.
# 500 = 14 000 images totales → chargement ~2 min, entraînement rapide.
MAX_IMAGES_PER_CLASS = 500

# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

# Nombre maximum d'époques (EarlyStopping arrêtera avant si nécessaire)
EPOCHS = 20

# Taux d'apprentissage initial pour Adam
# 1e-3 est le défaut conseillé par les auteurs d'Adam (Kingma & Ba, 2014)
LEARNING_RATE = 0.001    # Transfer learning : LR plus élevé pour la tête de classification

# ---------------------------------------------------------------------------
# Application temps réel
# ---------------------------------------------------------------------------

# Seuil de confiance minimale pour valider une prédiction
# En dessous de 85 %, on ignore la prédiction (trop incertaine)
CONFIDENCE_THRESHOLD = 0.97

# Délai minimal (en secondes) entre deux lettres acceptées
# Évite d'ajouter 30× la même lettre pendant qu'on tient le signe
LETTER_DELAY = 2.5

# ---------------------------------------------------------------------------
# Classes — les 28 classes du dataset madhavanair (Kaggle)
# ---------------------------------------------------------------------------
# Ce dataset contient 28 dossiers numérotés (0-27) remappés en :
#   0-25 → A-Z  |  26 → del  |  27 → nothing  (space absent de ce dataset)
# Triées alphabétiquement pour correspondre à l'ordre lexicographique des
# sous-dossiers créés par download_dataset.py (A, B, ..., Z, del, nothing)
CLASS_NAMES = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing'
])

# Nombre de classes (calculé dynamiquement pour ne pas se tromper)
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Création automatique des dossiers nécessaires
# ---------------------------------------------------------------------------
# Fait ici pour que tout import de config.py crée les dossiers si besoin
for _dir in [OUTPUTS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)
