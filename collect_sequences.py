import cv2
import mediapipe as mp
import numpy as np
import os
import json

# ── Konfigurasi ───────────────────────────────────────────────────────────────
DATA_DIR        = "data_kata"
SEQUENCE_LENGTH = 70          # jumlah frame per sampel
SAMPLES_PER_CLASS = 100       # sampel per kata (lebih sedikit dari huruf karena lebih kompleks)
NUM_FEATURES    = 126         # 2 tangan × 21 landmark × 3 (x, y, z)

# Daftar kata yang ingin dideteksi 
KATA = [
    "butuh", "bermain", "bantu", "maaf",
    "ya", "tidak", "terima_kasih", "halo",
    "saya", "kapan", "ayo", "dimana",
]

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

os.makedirs(DATA_DIR, exist_ok=True)


def extract_landmarks_dual(hand_landmarks_list, handedness_list):
    """Ekstrak 126 fitur dari 2 tangan, zero-pad jika hanya 1 tangan."""
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    for hl, hd in zip(hand_landmarks_list, handedness_list):
        label  = hd.classification[0].label
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32)
        coords -= coords[0]
        coords /= (np.max(np.abs(coords)) + 1e-6)
        flat = coords.flatten()
        if label == "Right":   # MediaPipe Right = tangan KIRI user
            left  = flat
        else:
            right = flat

    return np.concatenate([left, right])


def collect_for_word(cap, word):
    """
    Rekam SAMPLES_PER_CLASS sequence untuk satu kata.
    Setiap sequence = SEQUENCE_LENGTH frame landmark.
    """
    sequences  = []
    collecting = False
    current_seq = []

    print(f"\n[INFO] Siap merekam kata '{word}'")
    print(f"       Tekan SPACE untuk mulai merekam setiap sequence.")
    print(f"       Target: {SAMPLES_PER_CLASS} sequence @ {SEQUENCE_LENGTH} frame")

    while len(sequences) < SAMPLES_PER_CLASS:
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
                    cv2.putText(frame, display, (wx-20, wy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

        # Progress bar sequence
        if collecting:
            progress = len(current_seq) / SEQUENCE_LENGTH
            bar_w    = int(400 * progress)
            cv2.rectangle(frame, (140, frame.shape[0]-30), (540, frame.shape[0]-10), (40,40,40), -1)
            cv2.rectangle(frame, (140, frame.shape[0]-30), (140+bar_w, frame.shape[0]-10), (0,200,100), -1)
            cv2.putText(frame, f"Merekam: {len(current_seq)}/{SEQUENCE_LENGTH}",
                        (10, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,100), 2)

        # Status
        status_col = (0,200,100) if collecting else (0,165,255)
        cv2.putText(frame, f"Kata: '{word}'  |  Seq: {len(sequences)}/{SAMPLES_PER_CLASS}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_col, 2)
        cv2.putText(frame, f"Tangan: {num_hands}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if num_hands > 0 else (80,80,80), 2)
        if not collecting:
            cv2.putText(frame, "SPACE = mulai | Q = keluar", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        cv2.imshow("Collect Sequences - BISINDO Kata", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return sequences, True   # quit signal
        elif key == ord(" ") and not collecting:
            collecting  = True
            current_seq = []

        if collecting:
            # Ekstrak fitur frame ini (zeros jika tidak ada tangan)
            if result.multi_hand_landmarks:
                feat = extract_landmarks_dual(
                    result.multi_hand_landmarks,
                    result.multi_handedness,
                )
            else:
                feat = np.zeros(NUM_FEATURES, dtype=np.float32)

            current_seq.append(feat.tolist())

            if len(current_seq) == SEQUENCE_LENGTH:
                sequences.append(current_seq)
                current_seq = []
                collecting  = False
                print(f"  [OK] Sequence {len(sequences)}/{SAMPLES_PER_CLASS} terekam")

    return sequences, False


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak ditemukan!")

    save_path = os.path.join(DATA_DIR, "sequences.json")
    all_data  = {}
    if os.path.exists(save_path):
        with open(save_path) as f:
            all_data = json.load(f)
        print(f"[INFO] Data existing: {list(all_data.keys())}")

    print("=" * 55)
    print("  BISINDO KATA — SEQUENCE DATA COLLECTOR")
    print("=" * 55)
    print(f"Kata tersedia: {', '.join(KATA)}")
    print("Ketik kata yang ingin direkam, atau 'ALL' untuk semua.")
    target = input("Pilih kata: ").strip().lower()

    words_to_collect = KATA if target == "all" else [target]

    for word in words_to_collect:
        if word not in KATA:
            print(f"[WARN] Kata '{word}' tidak ada di daftar, skip.")
            continue
        seqs, quit_signal = collect_for_word(cap, word)
        if seqs:
            all_data[word] = all_data.get(word, []) + seqs
            print(f"[OK] '{word}': {len(all_data[word])} sequence")
            with open(save_path, "w") as f:
                json.dump(all_data, f)
        if quit_signal:
            break

    cap.release()
    cv2.destroyAllWindows()
    total_seq = sum(len(v) for v in all_data.values())
    print(f"\n[DONE] {len(all_data)} kata, {total_seq} total sequence → {save_path}")


if __name__ == "__main__":
    main()