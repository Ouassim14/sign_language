# =============================================================================
# src/train.py — Script principal d'entraînement du modèle ASL
# =============================================================================
# Ce script orchestre l'ensemble du pipeline : chargement des données,
# prétraitement, augmentation, construction du modèle, entraînement avec
# callbacks, puis évaluation complète des performances.

import os
import sys
import json

# Force UTF-8 sur stdout/stderr pour éviter UnicodeEncodeError sur Windows
# (le terminal cp1252 ne supporte pas les flèches → et les accents dans print())
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np

# Ajout du répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from src.preprocessing import load_dataset, split_dataset, visualize_samples
from src.augmentation  import get_datagen, visualize_augmentation
from src.model         import build_model
from src.evaluate      import (
    plot_training_curves,
    plot_confusion_matrix,
    print_classification_report,
    show_prediction_errors
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# =============================================================================
# Fonction principale d'entraînement
# =============================================================================

def train():
    """
    Pipeline complet : données → modèle → entraînement → évaluation.
    """

    print("\n" + "=" * 65)
    print("  ENTRAÎNEMENT DU MODÈLE ASL — RECONNAISSANCE DE L'ALPHABET")
    print("=" * 65 + "\n")

    # -------------------------------------------------------------------------
    # ÉTAPE 1 : Chargement du dataset
    # -------------------------------------------------------------------------
    print("[ÉTAPE 1/6] Chargement du dataset depuis :", config.DATASET_PATH)

    try:
        X, y, class_names = load_dataset(config.DATASET_PATH, config.IMG_SIZE)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Vérification de cohérence des classes
    if len(class_names) != config.NUM_CLASSES:
        print(f"[AVERTISSEMENT] {len(class_names)} classes trouvées dans le dataset, "
              f"mais {config.NUM_CLASSES} attendues dans config.py.")
        print(f"  Classes trouvées : {class_names}")

    # -------------------------------------------------------------------------
    # ÉTAPE 2 : Statistiques et visualisation des données
    # -------------------------------------------------------------------------
    print("\n[ÉTAPE 2/6] Statistiques du dataset")
    print(f"  - Images totales  : {len(X)}")
    print(f"  - Shape par image : {X[0].shape}")
    print(f"  - Nombre de classes : {len(class_names)}")
    print(f"  - Classes : {class_names}")

    # Distribution des classes
    print("\n  Distribution des classes :")
    for idx, name in enumerate(class_names):
        count = np.sum(y == idx)
        bar = "#" * (count // 100)
        print(f"    {name:10s} : {count:5d} images  {bar}")

    # Visualisation d'exemples (sauvegardée dans outputs/)
    visualize_samples(X, y, class_names, n=5)

    # -------------------------------------------------------------------------
    # ÉTAPE 3 : Split train / validation / test
    # -------------------------------------------------------------------------
    print("\n[ÉTAPE 3/6] Division du dataset en train / validation / test")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print(f"\n  Résumé des splits :")
    print(f"    X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"    X_val   : {X_val.shape}  |  y_val   : {y_val.shape}")
    print(f"    X_test  : {X_test.shape}  |  y_test  : {y_test.shape}")

    # -------------------------------------------------------------------------
    # ÉTAPE 4 : Augmentation de données
    # -------------------------------------------------------------------------
    print("\n[ÉTAPE 4/6] Configuration de l'augmentation de données")

    datagen = get_datagen()

    # Visualisation de l'augmentation sur un exemple
    sample_idx = np.random.randint(0, len(X_train))
    visualize_augmentation(X_train[sample_idx], datagen, n=8)

    # On n'utilise PAS datagen.flow() pour l'entraînement principal.
    # Bug TF 2.21 : flow(shuffle=True) mélange X et y indépendamment.
    # model.fit() gère lui-même le shuffle correctement (shuffle=True par défaut).
    # L'augmentation s'applique batch par batch via un generateur custom.
    print(f"\n  - Taille de batch    : {config.BATCH_SIZE}")
    print(f"  - Images train       : {len(X_train)}")

    # -------------------------------------------------------------------------
    # ÉTAPE 5 : Construction et configuration du modèle
    # -------------------------------------------------------------------------
    print("\n[ÉTAPE 5/6] Construction du modèle CNN")

    input_shape = (config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
    num_classes = len(class_names)

    model = build_model(input_shape, num_classes)

    # Callbacks — mécanismes de contrôle automatique de l'entraînement

    callbacks = [

        # EarlyStopping : arrête l'entraînement si val_accuracy ne s'améliore
        # plus depuis 5 époques. restore_best_weights=True recharge les meilleurs
        # poids au lieu des derniers (qui pourraient être overfit).
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # ModelCheckpoint : sauvegarde automatiquement le meilleur modèle.
        # Ne sauvegarde que si val_accuracy s'améliore (save_best_only=True).
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # ReduceLROnPlateau : divise le LR par 2 si val_loss stagne 3 époques.
        # min_lr empêche le LR de devenir infiniment petit (instabilité numérique).
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    # -------------------------------------------------------------------------
    # ÉTAPE 6 : Entraînement
    # -------------------------------------------------------------------------
    print("\n[ÉTAPE 6/6] Lancement de l'entraînement")
    print(f"  - Époques max  : {config.EPOCHS}")
    print(f"  - LR initial   : {config.LEARNING_RATE}")
    print(f"  - Modèle best  : {config.MODEL_PATH}")
    print()

    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        shuffle=True,       # Keras couple correctement X et y lors du shuffle
        verbose=1
    )

    # -------------------------------------------------------------------------
    # Sauvegarde de l'historique d'entraînement
    # -------------------------------------------------------------------------
    history_path = os.path.join(config.OUTPUTS_DIR, "training_history.json")
    # Conversion des valeurs numpy en float Python standard pour JSON
    history_serializable = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"\n[TRAIN] Historique d'entraînement sauvegardé → {history_path}")

    # -------------------------------------------------------------------------
    # ÉVALUATION FINALE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ÉVALUATION DU MODÈLE SUR LE TEST SET")
    print("=" * 65 + "\n")

    # Courbes d'entraînement
    plot_training_curves(history)

    # Matrice de confusion
    plot_confusion_matrix(model, X_test, y_test, class_names)

    # Rapport de classification complet
    print_classification_report(model, X_test, y_test, class_names)

    # Exemples d'erreurs
    show_prediction_errors(model, X_test, y_test, class_names, n=10)

    # Score final sur le test set
    print("\n[TRAIN] Évaluation finale sur X_test :")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss     : {test_loss:.4f}")
    print(f"  Accuracy : {test_acc:.4f} ({test_acc*100:.2f} %)")

    print("\n" + "=" * 65)
    print("  ENTRAÎNEMENT TERMINÉ !")
    print(f"  Modèle sauvegardé : {config.MODEL_PATH}")
    print(f"  Graphiques        : {config.OUTPUTS_DIR}")
    print("=" * 65)

    return model, history, class_names


# =============================================================================
# Point d'entrée principal
# =============================================================================

if __name__ == "__main__":
    train()
