import cv2
import mediapipe as mp
import numpy as np
import pickle, os, time
from collections import deque

# ── Cek TensorFlow ────────────────────────────────────────────────────────────
try:
    from tensorflow import keras
except ImportError:
    raise ImportError("Install TensorFlow: pip install tensorflow")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
MODEL_DIR       = "model_kata2"
SEQUENCE_LENGTH = 70
NUM_FEATURES    = 126
CONF_THRESHOLD  = 0.80    # prediksi ditampilkan hanya jika confidence > nilai ini
STABLE_FRAMES   = 8       # berapa frame harus stabil sebelum kata dikunci


def load_model():
    model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
    le_path    = os.path.join(MODEL_DIR, "label_encoder.pkl")
    for p in [model_path, le_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"'{p}' tidak ditemukan. Jalankan train_lstm.py dahulu."
            )
    model = keras.models.load_model(model_path)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    return model, le


# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6,
)


def extract_landmarks_dual(hand_landmarks_list, handedness_list):
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    for hl, hd in zip(hand_landmarks_list, handedness_list):
        label  = hd.classification[0].label
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32)
        coords -= coords[0]
        coords /= (np.max(np.abs(coords)) + 1e-6)
        flat = coords.flatten()
        if label == "Right":
            left  = flat
        else:
            right = flat
    return np.concatenate([left, right])


# ── Sliding Window Predictor ──────────────────────────────────────────────────
class WordPredictor:
    """
    Sliding window: buffer selalu menyimpan 30 frame terakhir.
    Prediksi dijalankan setiap frame (bukan hanya tiap 30 frame),
    sehingga lebih responsif.
    """
    def __init__(self, model, le, seq_len=SEQUENCE_LENGTH,
                 threshold=CONF_THRESHOLD, stable=STABLE_FRAMES):
        self.model     = model
        self.le        = le
        self.seq_len   = seq_len
        self.threshold = threshold
        self.stable    = stable
        self.buffer    = deque(maxlen=seq_len)   # sliding window
        self.recent    = deque(maxlen=stable)    # stabilizer
        self.locked_word = None
        self.locked_conf = 0.0
        self.last_added  = None                  # kata terakhir yang ditambah ke history

    def update(self, frame_features: np.ndarray):
        """Masukkan fitur 1 frame, kembalikan (word, confidence) atau (None, 0)."""
        self.buffer.append(frame_features)
        if len(self.buffer) < self.seq_len:
            return None, 0.0

        seq    = np.array(self.buffer, dtype=np.float32)[np.newaxis]  # (1, 30, 126)
        proba  = self.model.predict(seq, verbose=0)[0]
        idx    = np.argmax(proba)
        conf   = float(proba[idx])
        word   = self.le.inverse_transform([idx])[0] if conf >= self.threshold else None

        self.recent.append(word)

        # Kunci kata hanya jika STABLE_FRAMES terakhir konsisten
        counts = {}
        for w in self.recent:
            if w: counts[w] = counts.get(w, 0) + 1
        if counts:
            best = max(counts, key=counts.get)
            if counts[best] >= self.stable:
                self.locked_word = best
                self.locked_conf = conf
            else:
                self.locked_word = None
        else:
            self.locked_word = None

        return self.locked_word, self.locked_conf

    def reset(self):
        self.buffer.clear()
        self.recent.clear()
        self.locked_word = None


# ── UI helpers ────────────────────────────────────────────────────────────────
def draw_conf_bar(frame, x, y, w, h, conf, color):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40,40,40), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*conf), y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (80,80,80), 1)


def draw_sequence_bar(frame, buffer_len, seq_len):
    """Tampilkan progress buffer 30 frame di bagian bawah."""
    fh, fw = frame.shape[:2]
    bar_w  = fw - 40
    filled = int(bar_w * buffer_len / seq_len)
    cv2.rectangle(frame, (20, fh-18), (20+bar_w, fh-6), (30,30,30), -1)
    col = (0,200,100) if buffer_len == seq_len else (0,120,200)
    cv2.rectangle(frame, (20, fh-18), (20+filled, fh-6), col, -1)
    cv2.putText(frame, f"Buffer: {buffer_len}/{seq_len}",
                (20, fh-22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)


def overlay_ui(frame, word, conf, fps, history, num_hands, buffer_len, seq_len):
    h, w = frame.shape[:2]

    # Panel kiri atas
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (250,220), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    hand_col = (0,255,100) if num_hands==2 else (0,200,255) if num_hands==1 else (80,80,80)
    cv2.putText(frame, f"Tangan: {num_hands}", (20,42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_col, 2)

    if word:
        conf_col = (0,220,100) if conf>0.9 else (0,165,255) if conf>0.8 else (60,80,220)
        cv2.putText(frame, word, (20,130),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, conf_col, 3)
        cv2.putText(frame, f"{conf*100:.1f}%", (20,165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)
        draw_conf_bar(frame, 20,178, 210,10, conf, conf_col)
    else:
        cv2.putText(frame, "Menunggu...", (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (20,205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

    # History teks
    if history:
        text = " ".join(history[-15:])
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0,h-50), (w,h), (10,10,10), -1)
        cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, text, (15,h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    # Sequence buffer bar
    draw_sequence_bar(frame, buffer_len, seq_len)

    # Shortcut hint
    cv2.putText(frame, "[Enter] Tambah kata  [Space] Spasi  [C] Clear  [S] Screenshot  [Q] Keluar",
                (w-620, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90,90,90), 1)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, le = load_model()
    predictor  = WordPredictor(model, le)
    history    = []
    word, conf = None, 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak ditemukan!")

    os.makedirs("screenshots", exist_ok=True)
    shot_count = 0
    prev_time  = time.time()

    print("[INFO] BISINDO Kata Detector aktif. Tekan Q untuk keluar.")
    print(f"       Kata yang dikenali: {list(le.classes_)}")

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
            for idx, hl in enumerate(result.multi_hand_landmarks):
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )
                if result.multi_handedness:
                    hd_lbl  = result.multi_handedness[idx].classification[0].label
                    display = "KIRI" if hd_lbl == "Right" else "KANAN"
                    wx = int(hl.landmark[0].x * frame.shape[1])
                    wy = int(hl.landmark[0].y * frame.shape[0])
                    cv2.putText(frame, display, (wx-20, wy-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

            feat = extract_landmarks_dual(
                result.multi_hand_landmarks,
                result.multi_handedness,
            )
        else:
            feat = np.zeros(NUM_FEATURES, dtype=np.float32)

        word, conf = predictor.update(feat)

        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        frame = overlay_ui(
            frame, word, conf, fps, history,
            num_hands, len(predictor.buffer), SEQUENCE_LENGTH
        )
        cv2.imshow("BISINDO Kata Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 13 and word:           # Enter
            if word != predictor.last_added:
                history.append(word)
                predictor.last_added = word
                print(f"[OK] Ditambahkan: '{word}'")
        elif key == ord(" "):
            history.append(" ")
            predictor.last_added = None
        elif key == ord("c"):
            history.clear()
            predictor.reset()
            word, conf = None, 0.0
            print("[INFO] History cleared.")
        elif key == ord("s"):
            path = f"screenshots/shot_{shot_count:03d}.jpg"
            cv2.imwrite(path, frame)
            shot_count += 1
            print(f"[OK] Screenshot: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Kalimat: {' '.join(str(x) for x in history)}")


if __name__ == "__main__":
    main()