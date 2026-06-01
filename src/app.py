# =============================================================================
# src/app.py — Application temps réel de détection ASL via webcam
# =============================================================================

import os
import sys
import time
import threading
import queue

try:
    import win32api
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("[AVERTISSEMENT] pyttsx3 non installé — synthèse vocale désactivée.")
    print("  Installez avec : pip install pyttsx3")
    TTS_AVAILABLE = False

from tensorflow.keras.models import load_model

# États de la machine d'état
STATE_WAITING    = 'WAITING'
STATE_CONFIRMING = 'CONFIRMING'

tts_queue = queue.Queue()


def tts_worker(engine):
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        tts_queue.task_done()


def say_async(text):
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            tts_queue.task_done()
        except Exception:
            pass
    tts_queue.put(text)


def say_word(text):
    tts_queue.put(text)


# =============================================================================
# Initialisation du moteur de synthèse vocale
# =============================================================================

def init_tts():
    if not TTS_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        print("[APP] Moteur de synthèse vocale initialisé.")
        return engine
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible d'initialiser pyttsx3 : {e}")
        return None


# =============================================================================
# Prétraitement de la ROI
# =============================================================================

def preprocess_roi(roi, img_size=config.IMG_SIZE, use_mask=True):
    # Masque HSV — uniquement pour la fenêtre debug, jamais appliqué à la prédiction
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Pipeline identique à l'entraînement : roi brut, crop carré centré, resize, norm
    h, w = roi.shape[:2]
    side = min(h, w)
    roi_cropped = roi[(h - side) // 2:(h - side) // 2 + side,
                      (w - side) // 2:(w - side) // 2 + side]
    roi_resized = cv2.resize(roi_cropped, (128, 128))
    roi_norm    = roi_resized.astype(np.float32) / 255.0
    tensor      = roi_norm.reshape(1, 128, 128, 3)
    return tensor, mask


# =============================================================================
# Affichage overlay
# =============================================================================

def draw_overlay(frame, sign_label, confidence, current_word, roi_coords,
                 stable_count=0, stable_required=5, use_mask=True,
                 state=STATE_WAITING, candidate_letter='', blink_on=True):

    x1, y1, x2, y2 = roi_coords
    h, w = frame.shape[:2]

    # ---- Rectangle ROI : vert en attente, jaune en confirmation ----
    if state == STATE_CONFIRMING:
        roi_color = (0, 255, 255) if blink_on else (0, 160, 160)
    else:
        roi_color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 3)

    # ---- Label état au-dessus de la ROI ----
    zone_label = "EN ATTENTE..." if state == STATE_WAITING else "CONFIRMER ?"
    zone_color = (0, 255, 0) if state == STATE_WAITING else (0, 255, 255)
    cv2.putText(frame, zone_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone_color, 2)

    # ---- Indicateur Masque (coin supérieur droit) ----
    mask_label = "Masque: ON" if use_mask else "Masque: OFF"
    mask_color = (0, 255, 180) if use_mask else (0, 100, 255)
    cv2.putText(frame, mask_label, (w - 160, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mask_color, 2)

    # ---- Prédiction courante (haut gauche) ----
    conf_pct   = f"{confidence * 100:.1f}%"
    pred_text  = f"Signe: {sign_label} ({conf_pct})"
    text_color = (0, 255, 0) if confidence >= config.CONFIDENCE_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, pred_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    if state == STATE_WAITING:
        # ---- Barre de stabilité (rouge → verte) ----
        ratio      = min(stable_count / stable_required, 1.0)
        bar_x      = 10
        bar_y      = 55
        bar_full_w = 200
        bar_h      = 18
        filled_w   = int(bar_full_w * ratio)
        bar_color  = (0, int(255 * ratio), int(255 * (1 - ratio)))

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_full_w, bar_y + bar_h), (80, 80, 80), -1)
        if filled_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_full_w, bar_y + bar_h), (200, 200, 200), 1)

        stab_text  = f"Stabilite: {stable_count}/{stable_required}"
        stab_color = (0, 255, 0) if stable_count >= stable_required else (0, 180, 255)
        cv2.putText(frame, stab_text, (bar_x + bar_full_w + 8, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, stab_color, 1)

        # ---- Lettre candidate en grand au centre de la ROI ----
        if candidate_letter and candidate_letter not in ('nothing', ''):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            letter_color = (0, 255, 0) if stable_count >= stable_required else (0, 200, 255)
            cv2.putText(frame, candidate_letter, (cx - 38, cy + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8)
            cv2.putText(frame, candidate_letter, (cx - 38, cy + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, letter_color, 4)

    else:  # STATE_CONFIRMING
        # ---- Lettre candidate clignotante en jaune (grande) ----
        if blink_on and candidate_letter:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.putText(frame, candidate_letter, (cx - 38, cy + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8)
            cv2.putText(frame, candidate_letter, (cx - 38, cy + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)

        # ---- Instructions de confirmation ----
        cv2.putText(frame, "ENTREE  = confirmer", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        cv2.putText(frame, "BACKSPACE = refuser", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 100, 255), 2)

    # ---- Fond semi-transparent en bas ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Mot: {current_word}_", (10, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

    cv2.putText(frame,
                "ENTREE=confirmer | BACKSPACE=refuser | S=dire mot | C=effacer | M=masque | Q=quitter",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    return frame


# =============================================================================
# Boucle principale
# =============================================================================

def run_app():
    if not os.path.exists(config.MODEL_PATH):
        print(f"[ERREUR] Modèle introuvable : '{config.MODEL_PATH}'")
        print("  Lancez d'abord l'entraînement avec : python src/train.py")
        sys.exit(1)

    print(f"[APP] Chargement du modèle depuis : {config.MODEL_PATH}")
    model = load_model(config.MODEL_PATH)
    print("[APP] Modèle chargé.")

    class_names = config.CLASS_NAMES
    print(f"[APP] Classes : {class_names}")

    engine = init_tts()
    if engine:
        tts_thread = threading.Thread(target=tts_worker, args=(engine,), daemon=True)
        tts_thread.start()

    print("[APP] Ouverture de la webcam (index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[APP] Webcam ouverte.")

    # ---- Variables d'état ----
    current_word     = ""
    state            = STATE_WAITING
    candidate_letter = ""

    stable_letter          = ""
    stable_count           = 0
    STABLE_FRAMES_REQUIRED = 5

    blink_on        = True
    last_blink_time = 0.0
    BLINK_INTERVAL  = 0.4

    use_mask       = True
    s_held         = False
    enter_held     = False
    backspace_held = False

    roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 400, 400

    WIN_NAME = 'ASL Sign Language Detector'
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Impossible de lire la frame.")
            break

        frame = cv2.flip(frame, 1)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            continue

        tensor, mask = preprocess_roi(roi, use_mask=use_mask)

        mask_display = cv2.resize(mask, (200, 200))
        cv2.imshow("Masque peau (M=toggle)", mask_display)

        predictions   = model(tensor, training=False).numpy()[0]
        predicted_idx = int(np.argmax(predictions))
        confidence    = float(predictions[predicted_idx])
        sign_label    = class_names[predicted_idx]

        current_time = time.time()

        # ================================================================
        # ÉTAPE 1 : Touches — avant la machine d'état pour que le reset
        #           soit effectif AVANT le prochain calcul de candidate.
        # ================================================================
        key = cv2.waitKey(1) & 0xFF

        if WIN32_AVAILABLE:
            enter_down     = bool(win32api.GetAsyncKeyState(0x0D) & 0x8000)
            backspace_down = bool(win32api.GetAsyncKeyState(0x08) & 0x8000)
            s_down         = bool(win32api.GetAsyncKeyState(ord('S')) & 0x8000)

            if enter_down and not enter_held:
                enter_held = True
                if state == STATE_CONFIRMING and candidate_letter:
                    if candidate_letter == 'del':
                        if current_word:
                            current_word = current_word[:-1]
                        print(f"[APP] DEL confirme. Mot : '{current_word}'")
                    elif candidate_letter != 'nothing':
                        current_word += candidate_letter
                        print(f"[APP] '{candidate_letter}' confirme. Mot : '{current_word}'")
                    state            = STATE_WAITING
                    stable_letter    = ""
                    stable_count     = 0
                    candidate_letter = ""
            elif not enter_down:
                enter_held = False

            if backspace_down and not backspace_held:
                backspace_held = True
                if state == STATE_CONFIRMING:
                    print(f"[APP] Candidat '{candidate_letter}' refuse.")
                    state            = STATE_WAITING
                    stable_letter    = ""
                    stable_count     = 0
                    candidate_letter = ""
                elif current_word:
                    current_word = current_word[:-1]
                    print(f"[APP] Suppression. Mot : '{current_word}'")
            elif not backspace_down:
                backspace_held = False

            if s_down and not s_held:
                s_held = True
                print(f"[DEBUG S] word='{current_word}' strip='{current_word.strip()}'")
                if current_word.strip():
                    say_word(current_word.strip())
                    print(f"[APP] Mot dit : '{current_word.strip()}'")
            elif not s_down:
                s_held = False

        # Touches via waitKey (Q, C, M, ESPACE toujours ; ENTER/BACKSPACE fallback)
        if key == ord('q') or key == ord('Q'):
            print("[APP] Fermeture demandee par l'utilisateur.")
            break
        elif key == 13 and not WIN32_AVAILABLE:
            if state == STATE_CONFIRMING and candidate_letter:
                if candidate_letter == 'del':
                    if current_word:
                        current_word = current_word[:-1]
                elif candidate_letter != 'nothing':
                    current_word += candidate_letter
                    print(f"[APP] '{candidate_letter}' confirme. Mot : '{current_word}'")
                state            = STATE_WAITING
                stable_letter    = ""
                stable_count     = 0
                candidate_letter = ""
        elif key == 8 and not WIN32_AVAILABLE:
            if state == STATE_CONFIRMING:
                state = STATE_WAITING; stable_letter = ""; stable_count = 0; candidate_letter = ""
            elif current_word:
                current_word = current_word[:-1]
        elif key == ord('c') or key == ord('C'):
            print(f"[APP] Mot efface. Etait : '{current_word}'")
            current_word = ""; state = STATE_WAITING
            stable_letter = ""; stable_count = 0; candidate_letter = ""
        elif key == ord('m') or key == ord('M'):
            use_mask = not use_mask
            stable_letter = ""; stable_count = 0
            if state == STATE_CONFIRMING:
                state = STATE_WAITING; candidate_letter = ""
            print(f"[APP] Masque peau : {'ON' if use_mask else 'OFF'}")
        elif key == ord(' '):
            current_word += ' '
            print(f"[APP] Espace ajoute. Mot : '{current_word}'")
            if engine:
                try: say_async('space')
                except Exception: pass

        # ================================================================
        # ÉTAPE 2 : Machine d'état — après le reset éventuel des touches
        # ================================================================
        if state == STATE_WAITING:
            if sign_label == 'nothing':
                stable_letter    = ""
                stable_count     = 0
                candidate_letter = ""
            elif sign_label == stable_letter:
                stable_count = min(stable_count + 1, STABLE_FRAMES_REQUIRED)
            else:
                stable_letter = sign_label
                stable_count  = 1

            # N'affiche le candidat qu'à partir de 2 frames stables (évite le flash post-confirmation)
            candidate_letter = stable_letter if stable_count >= 2 else ""

            if (stable_count >= STABLE_FRAMES_REQUIRED
                    and confidence >= config.CONFIDENCE_THRESHOLD
                    and sign_label not in ('nothing',)):
                state           = STATE_CONFIRMING
                candidate_letter = stable_letter
                blink_on        = True
                last_blink_time = current_time
                if engine:
                    try: say_async(candidate_letter)
                    except Exception: pass
                print(f"[APP] CONFIRMING -> candidate : '{candidate_letter}'")

        else:  # STATE_CONFIRMING
            if current_time - last_blink_time >= BLINK_INTERVAL:
                blink_on        = not blink_on
                last_blink_time = current_time

        # ================================================================
        # ÉTAPE 3 : Affichage — avec l'état déjà mis à jour
        # ================================================================
        frame = draw_overlay(
            frame, sign_label, confidence, current_word,
            (roi_x1, roi_y1, roi_x2, roi_y2),
            stable_count=stable_count,
            stable_required=STABLE_FRAMES_REQUIRED,
            use_mask=use_mask,
            state=state,
            candidate_letter=candidate_letter,
            blink_on=blink_on,
        )

        cv2.imshow(WIN_NAME, frame)
        cv2.setWindowTitle(WIN_NAME, 'ASL - Appuyez sur Q pour quitter')

    # ---- Nettoyage ----
    print("[APP] Libération des ressources...")
    cap.release()
    cv2.destroyAllWindows()
    if engine:
        try:
            engine.stop()
        except Exception:
            pass
    print("[APP] Application fermée proprement.")


# =============================================================================
# Point d'entrée principal
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  APPLICATION ASL — DÉTECTION EN TEMPS RÉEL")
    print("=" * 60)
    print()
    print("  Placez votre main dans le rectangle vert.")
    print(f"  Seuil de confiance : {config.CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"  Stabilité requise  : 5 frames consécutives")
    print()
    print("  CONTRÔLES :")
    print("    ENTRÉE    = confirmer la lettre candidate")
    print("    BACKSPACE = refuser le candidat / supprimer dernière lettre")
    print("    ESPACE    = ajouter un espace")
    print("    S         = lire le mot à voix haute")
    print("    C         = effacer tout le mot")
    print("    M         = activer/désactiver le masque peau")
    print("    Q         = quitter")
    print()
    run_app()
