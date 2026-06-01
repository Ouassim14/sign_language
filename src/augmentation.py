# =============================================================================
# src/augmentation.py — Augmentation de données pour améliorer la robustesse
# =============================================================================
# L'augmentation simule des variations naturelles (angle, éclairage, position)
# que la main peut avoir devant la caméra. Elle agit comme un régularisateur
# implicite qui réduit l'overfitting sans collecter plus de données.

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# =============================================================================
# Fonction 1 : Construction du générateur d'augmentation
# =============================================================================

def get_datagen():
    """
    Retourne un ImageDataGenerator configuré pour l'ASL.

    Choix des paramètres :
    - rotation_range=15       : les mains peuvent être légèrement inclinées
    - width/height_shift=0.1  : la main n'est pas toujours centrée dans la ROI
    - zoom_range=0.1          : distance variable de la main à la caméra
    - horizontal_flip=False   : CRITIQUE — en ASL les signes ne sont PAS
                                symétriques (ex. J et Z ont une trajectoire)
    - brightness_range=[0.8,1.2] : simule des conditions d'éclairage variables
    - fill_mode='nearest'     : remplit les pixels apparus après rotation/shift
                                avec le pixel le plus proche (naturel)
    """

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,      # NE PAS activer — cf. explication ci-dessus
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    print("[AUGMENTATION] ImageDataGenerator configuré :")
    print("  - Rotation         : ±15°")
    print("  - Décalage X/Y     : ±10 %")
    print("  - Zoom             : ±10 %")
    print("  - Flip horizontal  : DÉSACTIVÉ (signes asymétriques)")
    print("  - Luminosité       : [0.8, 1.2]")

    return datagen


# =============================================================================
# Fonction 2 : Visualisation des images augmentées
# =============================================================================

def visualize_augmentation(X_sample, datagen, n=8):
    """
    Prend une image de référence et génère n variantes augmentées.
    Affiche l'originale + les variantes côte à côte.
    Sauvegarde dans outputs/augmentation_examples.png.

    Paramètres :
        X_sample : une seule image, shape (H, W, 1) ou (1, H, W, 1)
        datagen  : ImageDataGenerator déjà configuré
        n        : nombre de variantes augmentées à générer
    """

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUTS_DIR, "augmentation_examples.png")

    # Assure que l'image est en (1, H, W, 1) pour le générateur
    if X_sample.ndim == 3:
        img_batch = np.expand_dims(X_sample, axis=0)
    else:
        img_batch = X_sample

    # Génère les variantes via le flux infini du générateur
    augmented_images = []
    gen = datagen.flow(img_batch, batch_size=1, shuffle=True, seed=42)
    for _ in range(n):
        aug_batch = next(gen)
        augmented_images.append(aug_batch[0])   # shape (H, W, 1)

    # Création de la figure : 1 colonne originale + n colonnes augmentées
    total_cols = 1 + n
    fig, axes = plt.subplots(1, total_cols, figsize=(total_cols * 2, 3))
    fig.suptitle("Augmentation de données ASL", fontsize=13)

    # Image originale (colonne 0)
    axes[0].imshow(img_batch[0].squeeze(), cmap='gray')
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')

    # Variantes augmentées (colonnes 1..n)
    for i, aug_img in enumerate(augmented_images):
        axes[i + 1].imshow(aug_img.squeeze(), cmap='gray')
        axes[i + 1].set_title(f"Aug {i+1}", fontsize=10)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[AUGMENTATION] Exemples d'augmentation sauvegardés → {output_path}")


# =============================================================================
# Point d'entrée pour test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test standalone : augmentation.py ===\n")

    # Crée une image synthétique (bruit blanc) pour tester sans dataset
    dummy_image = np.random.rand(config.IMG_SIZE, config.IMG_SIZE, 1).astype(np.float32)
    print(f"[TEST] Image de test générée : shape {dummy_image.shape}")

    datagen = get_datagen()
    visualize_augmentation(dummy_image, datagen, n=8)

    print("\n[OK] augmentation.py s'est exécuté sans erreur.")
