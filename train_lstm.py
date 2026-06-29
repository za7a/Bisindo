import json, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Konfigurasi ───────────────────────────────────────────────────────────────
DATA_PATH       = "data_kata/sequences.json"
MODEL_DIR       = "model_kata2"
SEQUENCE_LENGTH = 70
NUM_FEATURES    = 126
EPOCHS          = 100
BATCH_SIZE      = 32

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path):
    """Load sequences.json → X shape (N, 70, 126), y shape (N,)"""
    with open(path) as f:
        raw = json.load(f)

    X, y = [], []
    for label, sequences in raw.items():
        for seq in sequences:
            X.append(seq)   # shape: (70, 126)
            y.append(label)

    X = np.array(X, dtype=np.float32)   # (N, 70, 126)
    y = np.array(y)
    print(f"[INFO] X shape: {X.shape}  |  Kelas: {sorted(set(y))}")
    return X, y


def build_lstm_model(num_classes, seq_len=SEQUENCE_LENGTH, num_feat=NUM_FEATURES):
    """
    Arsitektur LSTM untuk deteksi kata dari sequence landmark.

    Input:  (batch, 70, 126)
    Output: (batch, num_classes) — probabilitas tiap kata
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_len, num_feat)),

        # Layer 1: Bidirectional LSTM — tangkap pola maju & mundur
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),

        # Layer 2: LSTM biasa
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        # Dense layers untuk klasifikasi
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),

        # Output: softmax → probabilitas tiap kata
        layers.Dense(num_classes, activation="softmax"),
    ], name="bisindo_kata_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Training history → {save_path}")


def plot_confusion(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — BISINDO Kata")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    try:
        import tensorflow as tf
        version = getattr(tf, '__version__', None) or getattr(tf.version, 'VERSION', 'unknown')
        print(f"[INFO] TensorFlow {version}")
    except ImportError:
        raise ImportError(
            "TensorFlow belum terinstall.\n"
            "Install dengan: pip install tensorflow==2.15.0"
        )

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data tidak ditemukan di '{DATA_PATH}'.\n"
            "Jalankan collect_sequences.py terlebih dahulu."
        )

    # ── Load & encode ──────────────────────────────────────────────────────
    X, y_raw = load_data(DATA_PATH)
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"[INFO] Jumlah kelas: {num_classes} → {list(le.classes_)}")

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Build & train ──────────────────────────────────────────────────────
    model = build_lstm_model(num_classes)
    model.summary()

    from tensorflow import keras
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=7,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "lstm_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\n[INFO] Mulai training ({EPOCHS} epoch max)...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluasi ───────────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    acc = (y_pred == y_test).mean()
    print(f"\n[RESULT] Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0,
    ))

    # ── Simpan ────────────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))
    plot_confusion(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        list(le.classes_),
        os.path.join(MODEL_DIR, "confusion_matrix.png"),
    )

    print(f"\n[DONE] Semua file tersimpan di '{MODEL_DIR}/'")
    print("       Jalankan inference_kata.py untuk mencoba!")


if __name__ == "__main__":
    main()