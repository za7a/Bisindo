import cv2
import mediapipe as mp
import numpy as np
import os
import json

# ── Konfigurasi ───────────────────────────────────────────────────────────────
DATA_DIR         = "data"
SAMPLES_PER_CLASS = 200
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(65, 91)]  # 0-9, A-Z

# Fitur = 2 tangan × 21 landmark × 3 (x,y,z) = 126 fitur
# Jika hanya 1 tangan terdeteksi, tangan yang tidak ada akan di-zero-pad
NUM_FEATURES = 126

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # ← deteksi hingga 2 tangan
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

os.makedirs(DATA_DIR, exist_ok=True)


def extract_landmarks_dual(hand_landmarks_list, handedness_list):
    """ 
    Ekstrak landmark dari 1 atau 2 tangan.
    
    Output selalu 126 fitur:
        [0:63]   = tangan KIRI  (zero jika tidak terdeteksi)
        [63:126] = tangan KANAN (zero jika tidak terdeteksi)
    
    Normalisasi: relatif ke wrist masing-masing tangan, lalu di-scale.
    """
    left_features  = np.zeros(63, dtype=np.float32)
    right_features = np.zeros(63, dtype=np.float32)

    for hl, hd in zip(hand_landmarks_list, handedness_list):
        label = hd.classification[0].label  # "Left" atau "Right"
        # "Left" dari MediaPipe = tangan KANAN user, dan sebaliknya
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark],
                          dtype=np.float32)
        # Normalisasi relatif ke wrist (landmark 0)
        coords -= coords[0]
        scale = np.max(np.abs(coords)) + 1e-6
        coords /= scale
        flat = coords.flatten()

        if label == "Right":          # MediaPipe Right = tangan KIRI user
            left_features = flat
        else:                         # MediaPipe Left  = tangan KANAN user
            right_features = flat

    return np.concatenate([left_features, right_features]).tolist()


def collect_for_class(cap, class_label):
    """Rekam SAMPLES_PER_CLASS sampel untuk satu kelas."""
    samples    = []
    collecting = False
    count      = 0

    print(f"\n[INFO] Siap merekam kelas '{class_label}'. Tekan SPACE untuk mulai.")

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
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )
            # Label L/R per tangan
            if result.multi_handedness:
                for idx, hd in enumerate(result.multi_handedness):
                    hl  = result.multi_hand_landmarks[idx]
                    lbl = hd.classification[0].label
                    # Titik wrist untuk label posisi
                    wx = int(hl.landmark[0].x * frame.shape[1])
                    wy = int(hl.landmark[0].y * frame.shape[0])
                    display = "KIRI" if lbl == "Right" else "KANAN"
                    cv2.putText(frame, display, (wx - 20, wy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)

        # Indikator jumlah tangan
        hand_color = (0, 255, 0) if num_hands > 0 else (0, 80, 220)
        hand_info  = f"Tangan terdeteksi: {num_hands}"
        cv2.putText(frame, hand_info, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2)

        # Status
        status = f"Kelas: {class_label}  |  Sampel: {count}/{SAMPLES_PER_CLASS}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if collecting else (0, 165, 255), 2)
        if not collecting:
            cv2.putText(frame, "Tekan SPACE untuk mulai", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Collect Data - BISINDO", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            collecting = True
        elif key == ord("q"):
            return samples, True   # quit signal

        if collecting and result.multi_hand_landmarks:
            features = extract_landmarks_dual(
                result.multi_hand_landmarks,
                result.multi_handedness,
            )
            samples.append(features)
            count += 1
            if count >= SAMPLES_PER_CLASS:
                break

    return samples, False


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak ditemukan!")

    all_data  = {}
    save_path = os.path.join(DATA_DIR, "landmarks.json")

    if os.path.exists(save_path):
        with open(save_path) as f:
            all_data = json.load(f)
        print(f"[INFO] Data existing: {sorted(all_data.keys())}")

    print("=" * 55)
    print("  BISINDO DATA COLLECTOR  ")
    print("=" * 55)
    print(f"Kelas: {', '.join(CLASSES)}")
    print("Ketik kelas (contoh: A / 5 / ALL untuk semua).")
    target = input("Pilih kelas: ").strip().upper()

    classes_to_collect = CLASSES if target == "ALL" else [target]

    for cls in classes_to_collect:
        if cls not in CLASSES:
            print(f"[WARN] Kelas '{cls}' tidak valid, skip.")
            continue
        samples, quit_signal = collect_for_class(cap, cls)
        if samples:
            all_data[cls] = all_data.get(cls, []) + samples
            print(f"[OK] '{cls}': {len(all_data[cls])} total sampel")
            with open(save_path, "w") as f:
                json.dump(all_data, f)
        if quit_signal:
            break

    cap.release()
    cv2.destroyAllWindows()
    total = sum(len(v) for v in all_data.values())
    print(f"\n[DONE] {len(all_data)} kelas, {total} total sampel → {save_path}")


if __name__ == "__main__":
    main()
