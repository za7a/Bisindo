import cv2
import mediapipe as mp
import numpy as np
import pickle, os, time
from collections import deque

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_DIR = "model"

def load_model():
    files = ["classifier.pkl", "label_encoder.pkl", "scaler.pkl"]
    for f in files:
        p = os.path.join(MODEL_DIR, f)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"'{p}' tidak ditemukan. Jalankan train_model.py dahulu."
            )
    with open(os.path.join(MODEL_DIR, "classifier.pkl"),    "rb") as f: model  = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f: le     = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),        "rb") as f: scaler = pickle.load(f)
    return model, le, scaler


# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # ← 2 tangan untuk BISINDO
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6,
)


def extract_landmarks_dual(hand_landmarks_list, handedness_list):
    """
    Ekstrak 126 fitur dari maksimal 2 tangan.
    [0:63]   = tangan KIRI  (zeros jika tidak ada)
    [63:126] = tangan KANAN (zeros jika tidak ada)
    """
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    for hl, hd in zip(hand_landmarks_list, handedness_list):
        label  = hd.classification[0].label   # "Left" / "Right" (sudah di-flip MediaPipe)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32)
        coords -= coords[0]
        coords /= (np.max(np.abs(coords)) + 1e-6)
        flat = coords.flatten()
        if label == "Right":   # MediaPipe Right = tangan KIRI user (mirror)
            left  = flat
        else:
            right = flat

    return np.concatenate([left, right])


# ── Smoother ──────────────────────────────────────────────────────────────────
class PredictionSmoother:
    def __init__(self, window=12):
        self.buf = deque(maxlen=window)

    def update(self, pred, conf):
        self.buf.append((pred, conf))

    def get(self):
        if not self.buf:
            return None, 0.0
        labels  = [p for p, _ in self.buf]
        majority = max(set(labels), key=labels.count)
        avg_conf = np.mean([c for l, c in self.buf if l == majority])
        return majority, avg_conf

    def clear(self):
        self.buf.clear()


# ── UI helpers ────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_conf_bar(img, x, y, w, h, conf, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + int(w * conf), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)


def overlay_ui(frame, label, confidence, fps, history, num_hands):
    h, w = frame.shape[:2]

    # ── Panel prediksi (kiri atas) ────────────────────────────────────────
    draw_rounded_rect(frame, 10, 10, 240, 210, 12, (15, 15, 15))

    hand_color = (0, 255, 100) if num_hands == 2 else \
                 (0, 200, 255) if num_hands == 1 else (80, 80, 80)

    cv2.putText(frame, f"Tangan: {num_hands}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2)

    if label and num_hands > 0:
        conf_color = (
            (0, 220, 100) if confidence > 0.8 else
            (0, 165, 255) if confidence > 0.5 else
            (60, 80, 220)
        )
        cv2.putText(frame, label, (50, 145),
                    cv2.FONT_HERSHEY_DUPLEX, 3.8, conf_color, 6)
        cv2.putText(frame, f"{confidence*100:.1f}%", (25, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        draw_conf_bar(frame, 20, 185, 200, 12, confidence, conf_color)
    else:
        cv2.putText(frame, "Tidak ada tangan", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    # ── FPS ───────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.0f}", (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

    # ── Teks history (bawah) ──────────────────────────────────────────────
    if history:
        text  = "".join(history[-40:])
        bar_h = 50
        draw_rounded_rect(frame, 0, h - bar_h, w, h, 0, (10, 10, 10), alpha=0.8)
        cv2.putText(frame, text, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # ── Shortcut hint (kanan atas) ────────────────────────────────────────
    hint = "[Enter] Tambah  [Space] Spasi  [C] Clear  [S] Screenshot  [Q] Quit"
    cv2.putText(frame, hint, (w - 620, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, le, scaler = load_model()
    smoother  = PredictionSmoother(window=12)
    history   = []
    label, confidence = None, 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak ditemukan!")

    os.makedirs("screenshots", exist_ok=True)
    shot_count = 0
    prev_time  = time.time()

    print("[INFO] BISINDO Detector aktif. Tekan Q untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        num_hands = 0

        if result.multi_hand_landmarks:
            num_hands = len(result.multi_hand_landmarks)

            # Gambar landmark semua tangan
            for idx, hl in enumerate(result.multi_hand_landmarks):
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )
                # Label L/R di atas wrist
                if result.multi_handedness:
                    hd_label = result.multi_handedness[idx].classification[0].label
                    display  = "KIRI" if hd_label == "Right" else "KANAN"
                    wx = int(hl.landmark[0].x * frame.shape[1])
                    wy = int(hl.landmark[0].y * frame.shape[0])
                    cv2.putText(frame, display, (wx - 25, wy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Ekstrak 126 fitur & prediksi
            features        = extract_landmarks_dual(
                result.multi_hand_landmarks,
                result.multi_handedness,
            ).reshape(1, -1)
            features_scaled = scaler.transform(features)
            proba           = model.predict_proba(features_scaled)[0]
            pred_idx        = np.argmax(proba)
            pred_label      = le.inverse_transform([pred_idx])[0]
            pred_conf       = proba[pred_idx]

            smoother.update(pred_label, pred_conf)
            label, confidence = smoother.get()
        else:
            smoother.clear()

        # FPS
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        frame = overlay_ui(frame, label, confidence, fps, history, num_hands)
        cv2.imshow("BISINDO Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 13 and label:       # Enter
            history.append(label)
        elif key == ord(" "):
            history.append(" ")
        elif key == ord("c"):
            history.clear()
            print("[INFO] History cleared.")
        elif key == ord("s"):
            path = f"screenshots/shot_{shot_count:03d}.jpg"
            cv2.imwrite(path, frame)
            shot_count += 1
            print(f"[OK] Screenshot: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Teks: {''.join(history)}")


if __name__ == "__main__":
    main()
