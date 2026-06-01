# =============================================================================
# src/evaluate.py — Évaluation et visualisation des performances du modèle
# =============================================================================
# Ce module produit tous les graphiques d'analyse post-entraînement :
# courbes d'apprentissage, matrice de confusion, rapport de classification,
# et exemples d'erreurs. Indispensable pour diagnostiquer les faiblesses du modèle.

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# =============================================================================
# Fonction 1 : Courbes d'entraînement (accuracy et loss)
# =============================================================================

def plot_training_curves(history):
    """
    Trace les courbes train vs validation pour accuracy et loss.

    La ligne verticale pointillée marque l'époque du meilleur val_accuracy,
    ce qui permet de voir si EarlyStopping a arrêté au bon moment.

    Paramètre :
        history : objet History retourné par model.fit(), ou dict équivalent
    """

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUTS_DIR, "training_curves.png")

    # Supporte à la fois l'objet History de Keras et un dict chargé depuis JSON
    if hasattr(history, 'history'):
        hist = history.history
    else:
        hist = history

    epochs_range = range(1, len(hist['accuracy']) + 1)

    # Époque du meilleur val_accuracy (pour la ligne verticale)
    best_epoch = int(np.argmax(hist['val_accuracy'])) + 1
    best_val_acc = max(hist['val_accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'entraînement — Modèle ASL CNN", fontsize=14)

    # ---- Subplot 1 : Accuracy ----
    ax1.plot(epochs_range, hist['accuracy'],     label='Train Accuracy',      color='royalblue',  linewidth=2)
    ax1.plot(epochs_range, hist['val_accuracy'], label='Val Accuracy',        color='darkorange',  linewidth=2)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5,
                label=f'Meilleur val_acc : époque {best_epoch} ({best_val_acc:.4f})')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy — Train vs Validation')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # ---- Subplot 2 : Loss ----
    ax2.plot(epochs_range, hist['loss'],     label='Train Loss',      color='royalblue',  linewidth=2)
    ax2.plot(epochs_range, hist['val_loss'], label='Val Loss',        color='darkorange',  linewidth=2)
    ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5,
                label=f'Meilleur val_acc : époque {best_epoch}')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss — Train vs Validation')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"[EVALUATE] Courbes d'entraînement sauvegardées → {output_path}")
    print(f"           Meilleur val_accuracy : {best_val_acc:.4f} à l'époque {best_epoch}")


# =============================================================================
# Fonction 2 : Matrice de confusion
# =============================================================================

def plot_confusion_matrix(model, X_test, y_test, class_names):
    """
    Calcule et affiche la matrice de confusion sur l'ensemble de test.

    La matrice de confusion révèle quelles classes sont confondues entre elles,
    ce qui est plus informatif que la simple accuracy globale. Par exemple,
    B et D peuvent être visuellement proches en ASL.

    Paramètres :
        model       : modèle Keras entraîné
        X_test      : images de test, shape (N, H, W, 1)
        y_test      : labels one-hot, shape (N, num_classes)
        class_names : liste des noms de classes
    """

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")

    print("\n[EVALUATE] Calcul de la matrice de confusion...")

    # Prédictions du modèle (indices des classes prédites)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Labels réels (depuis one-hot vers entiers)
    y_true = np.argmax(y_test, axis=1)

    # Matrice de confusion avec sklearn
    cm = confusion_matrix(y_true, y_pred)

    # Normalisation par ligne pour avoir des pourcentages
    # Évite que les classes avec plus d'images dominent visuellement
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor='lightgray'
    )
    ax.set_xlabel('Classe Prédite', fontsize=12)
    ax.set_ylabel('Classe Réelle', fontsize=12)
    ax.set_title('Matrice de Confusion Normalisée — ASL CNN', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    # Accuracy globale sur le test set
    test_accuracy = np.mean(y_pred == y_true)
    print(f"[EVALUATE] Matrice de confusion sauvegardée → {output_path}")
    print(f"[EVALUATE] Accuracy sur le test set : {test_accuracy:.4f} ({test_accuracy*100:.2f} %)")

    return y_pred, y_true


# =============================================================================
# Fonction 3 : Rapport de classification complet
# =============================================================================

def print_classification_report(model, X_test, y_test, class_names):
    """
    Affiche le rapport sklearn complet : précision, rappel, F1 par classe.

    La précision mesure "quand je prédit X, j'ai raison à X %".
    Le rappel mesure "sur tous les vrais X, j'en détecte X %".
    Le F1 est leur moyenne harmonique — utile quand précision et rappel divergent.
    """

    print("\n[EVALUATE] Génération du rapport de classification...")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4               # 4 décimales pour plus de précision
    )

    print("\n" + "=" * 70)
    print("RAPPORT DE CLASSIFICATION — ASL CNN")
    print("=" * 70)
    print(report)
    print("=" * 70)

    # Sauvegarde du rapport dans un fichier texte
    report_path = os.path.join(config.OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE CLASSIFICATION — ASL CNN\n")
        f.write("=" * 70 + "\n")
        f.write(report)
    print(f"[EVALUATE] Rapport sauvegardé → {report_path}")


# =============================================================================
# Fonction 4 : Visualisation des erreurs de prédiction
# =============================================================================

def show_prediction_errors(model, X_test, y_test, class_names, n=10):
    """
    Identifie et affiche les n premières erreurs du modèle.

    Voir les erreurs concrètes est souvent plus instructif que les métriques :
    on peut constater si les erreurs sont "raisonnables" (B confondu avec D)
    ou problématiques (A confondu avec Z = erreur très différente visuellement).

    Paramètres :
        n : nombre d'exemples d'erreurs à afficher
    """

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUTS_DIR, "prediction_errors.png")

    print(f"\n[EVALUATE] Recherche des {n} premières erreurs de prédiction...")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Indices des prédictions incorrectes
    error_indices = np.where(y_pred != y_true)[0]
    total_errors = len(error_indices)
    print(f"[EVALUATE] Erreurs totales : {total_errors} / {len(y_true)} ({total_errors/len(y_true)*100:.1f} %)")

    if total_errors == 0:
        print("[EVALUATE] Aucune erreur à afficher — modèle parfait sur le test set !")
        return

    # On limite aux n premières erreurs disponibles
    n_show = min(n, total_errors)
    selected_errors = error_indices[:n_show]

    # Calcul du nombre de colonnes/lignes pour la grille
    ncols = 5
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.2))
    fig.suptitle(f"Exemples d'erreurs de prédiction ({n_show} premières)", fontsize=13)
    axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, err_idx in enumerate(selected_errors):
        ax = axes_flat[i]
        ax.imshow(X_test[err_idx].squeeze(), cmap='gray')

        true_class  = class_names[y_true[err_idx]]
        pred_class  = class_names[y_pred[err_idx]]
        confidence  = y_pred_probs[err_idx][y_pred[err_idx]] * 100

        ax.set_title(
            f"Réel : {true_class}\nPrédit : {pred_class} ({confidence:.1f}%)",
            fontsize=9,
            color='red'
        )
        ax.axis('off')

    # Masquer les axes vides si n_show < nrows * ncols
    for j in range(n_show, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[EVALUATE] Erreurs de prédiction sauvegardées → {output_path}")


# =============================================================================
# Point d'entrée pour test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test standalone : evaluate.py ===\n")
    print("[INFO] Pour tester evaluate.py, lancez d'abord train.py pour générer un modèle.")
    print("       Ce module est appelé automatiquement à la fin de train.py.")

    # Test minimal : simuler un historique pour vérifier plot_training_curves
    import json
    history_path = os.path.join(config.OUTPUTS_DIR, "training_history.json")

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            hist = json.load(f)
        print(f"\n[TEST] Historique trouvé ({len(hist['accuracy'])} époques). Génération des courbes...")
        plot_training_curves(hist)
        print("[OK] evaluate.py test partiel réussi.")
    else:
        print(f"[INFO] Pas d'historique trouvé dans '{history_path}'. Lancez train.py d'abord.")
