# =============================================================================
# src/preprocessing.py — Chargement et préparation du dataset ASL
# =============================================================================
# Ce module transforme les images brutes du dataset Kaggle en tableaux NumPy
# normalisés, prêts pour l'entraînement d'un réseau convolutif.

import os
import sys
import json

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sans GUI pour éviter les erreurs headless
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Ajout du répertoire parent au PYTHONPATH pour importer config depuis n'importe où
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# =============================================================================
# Fonction 1 : Chargement du dataset depuis le disque
# =============================================================================

def load_dataset(dataset_path=config.DATASET_PATH, img_size=config.IMG_SIZE,
                 max_per_class=config.MAX_IMAGES_PER_CLASS,
                 channels=config.IMG_CHANNELS):
    """
    Charge le dataset ASL avec center-crop carré puis resize.

    Pipeline :
      1. cv2.imread → BGR couleur
      2. Center-crop carré (min(H,W) × min(H,W)) — focalise sur la main
      3. resize(img_size, img_size)
      4. Normalisation [0,1]
      5. Si channels=1 : conversion grayscale + ajout dim canal
         Si channels=3 : conserve BGR, shape (H,W,3)

    Pourquoi center-crop ?
    Les images sont 480×640 (webcam plein format). Sans crop, le resize 64×64
    moyenne 75 pixels par pixel de sortie — la main est illisible. Le crop
    carré conserve la zone centrale où la main est le plus souvent placée.

    Retourne :
        X          : np.array (N, img_size, img_size, channels), valeurs dans [0,1]
        y          : np.array d'entiers (labels)
        class_names: liste triée des noms de classes
    """

    # Vérification que le dossier existe avant de commencer
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"[ERREUR] Dataset introuvable : '{dataset_path}'\n"
            "Téléchargez le dataset sur Kaggle (madhavanair/american-sign-language-dataset)\n"
            "et décompressez-le dans le dossier 'data/asl_alphabet_train/'."
        )

    # Récupère et trie les classes (= noms des sous-dossiers)
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    if not class_names:
        raise ValueError(f"[ERREUR] Aucun sous-dossier de classe trouvé dans '{dataset_path}'.")

    print(f"\n[PREPROCESSING] {len(class_names)} classes détectées : {class_names}")

    images = []
    labels = []

    # Itère sur chaque classe avec une barre de progression globale
    for class_idx, class_name in enumerate(tqdm(class_names, desc="Chargement des classes")):
        class_dir = os.path.join(dataset_path, class_name)
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not image_files:
            print(f"  [AVERTISSEMENT] Classe '{class_name}' vide, ignorée.")
            continue

        # Limite le nombre d'images si max_per_class est défini
        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        # Charge chaque image de la classe
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Center-crop carré : min(H,W) × min(H,W)
            # Élimine les bandes noires latérales et recentre sur la main
            h, w = img.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            img = img[y0:y0+side, x0:x0+side]

            if channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img, (img_size, img_size))
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_final = np.expand_dims(img_normalized, axis=-1)
            else:
                # Garde les 3 canaux BGR — plus d'information qu'en grayscale
                img_resized = cv2.resize(img, (img_size, img_size))
                img_final = img_resized.astype(np.float32) / 255.0

            images.append(img_final)
            labels.append(class_idx)

    # Conversion en tableaux NumPy pour l'efficacité mémoire et de calcul
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print(f"\n[PREPROCESSING] Dataset chargé :")
    print(f"  - Nombre total d'images : {len(X)}")
    print(f"  - Shape X               : {X.shape}")
    print(f"  - Plage de valeurs      : [{X.min():.2f}, {X.max():.2f}]")
    print(f"  - Nombre de classes     : {len(class_names)}")

    return X, y, class_names


# =============================================================================
# Fonction 2 : Division en ensembles train / validation / test
# =============================================================================

def split_dataset(X, y, test_size=0.2, val_size=0.5):
    """
    Découpe stratifiée en trois ensembles : entraînement, validation, test.

    Stratégie : double split
      1) 80 % train  +  20 % temp   (stratifié sur y)
      2) temp → 50 % val  +  50 % test

    Résultat final : 80 % train | 10 % val | 10 % test
    La stratification garantit des distributions de classes identiques dans
    chaque sous-ensemble, crucial avec 29 classes déséquilibrées.

    Les labels y sont encodés en one-hot pour categorical_crossentropy.
    """

    num_classes = len(np.unique(y))
    print(f"\n[SPLIT] Division du dataset ({len(X)} images, {num_classes} classes)")

    # Premier split : train vs temporaire
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,        # Reproductibilité
        stratify=y              # Même distribution de classes dans chaque split
    )

    # Deuxième split : validation vs test (sur la portion temporaire)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=42,
        stratify=y_temp
    )

    print(f"  - Train      : {len(X_train)} images ({len(X_train)/len(X)*100:.1f} %)")
    print(f"  - Validation : {len(X_val)} images  ({len(X_val)/len(X)*100:.1f} %)")
    print(f"  - Test       : {len(X_test)} images  ({len(X_test)/len(X)*100:.1f} %)")

    # Encodage one-hot : entier → vecteur binaire de taille num_classes
    # Ex : classe 3 → [0, 0, 0, 1, 0, ..., 0]
    y_train_oh = to_categorical(y_train, num_classes=num_classes)
    y_val_oh   = to_categorical(y_val,   num_classes=num_classes)
    y_test_oh  = to_categorical(y_test,  num_classes=num_classes)

    return X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh


# =============================================================================
# Fonction 3 : Visualisation d'exemples du dataset
# =============================================================================

def visualize_samples(X, y, class_names, n=5):
    """
    Affiche une grille de n exemples pour les 5 premières classes.
    Sauvegarde le résultat dans outputs/sample_images.png.

    Utile pour vérifier visuellement que le chargement s'est bien passé
    et que les labels correspondent aux bonnes images.
    """

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUTS_DIR, "sample_images.png")

    # On ne montre que les 5 premières classes pour garder la figure lisible
    classes_to_show = class_names[:5]
    num_classes_shown = len(classes_to_show)

    fig, axes = plt.subplots(
        num_classes_shown, n,
        figsize=(n * 2, num_classes_shown * 2.2)
    )
    fig.suptitle("Exemples du dataset ASL (5 premières classes)", fontsize=14, y=1.02)

    for row_idx, class_name in enumerate(classes_to_show):
        # Indices de toutes les images appartenant à cette classe
        class_idx = class_names.index(class_name)
        indices = np.where(y == class_idx)[0]

        # Sélection aléatoire de n images
        selected = np.random.choice(indices, size=min(n, len(indices)), replace=False)

        for col_idx, img_idx in enumerate(selected):
            ax = axes[row_idx, col_idx]

            # L'image est stockée (H, W, 1) → on squeeze pour afficher en 2D
            ax.imshow(X[img_idx].squeeze(), cmap='gray')
            ax.axis('off')

            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=12, rotation=0, labelpad=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[PREPROCESSING] Grille d'exemples sauvegardée → {output_path}")


# =============================================================================
# Point d'entrée pour test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test standalone : preprocessing.py ===\n")

    try:
        X, y, class_names = load_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
        visualize_samples(X, y, class_names, n=5)
        print("\n[OK] preprocessing.py s'est exécuté sans erreur.")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
