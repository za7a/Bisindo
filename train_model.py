import json, os, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ── Konfigurasi ───────────────────────────────────────────────────────────────
DATA_PATH  = "data/landmarks.json"
MODEL_DIR  = "model"
MODEL_TYPE = "mlp"    

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    X, y = [], []
    for label, samples in raw.items():
        for s in samples:
            X.append(s)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def build_model(model_type):
    if model_type == "mlp":
        # Input 126 fitur (2 tangan), hidden layer lebih besar
        return MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=True,
        )
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
        )
    raise ValueError(f"Unknown model: {model_type}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — BISINDO Classifier")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix → {save_path}")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data tidak ditemukan di {DATA_PATH}. "
            "Jalankan collect_data.py terlebih dahulu."
        )

    print("[INFO] Loading data...")
    X, y_raw = load_data(DATA_PATH)
    print(f"       Sampel  : {len(X)}")
    print(f"       Kelas   : {sorted(set(y_raw))}")
    print(f"       Fitur   : {X.shape[1]}  (126 = 2 tangan × 63)")

    le      = LabelEncoder()
    y       = le.fit_transform(y_raw)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")

    print(f"\n[INFO] Training {MODEL_TYPE.upper()}...")
    model = build_model(MODEL_TYPE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = (y_pred == y_test).mean()
    print(f"\n[RESULT] Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")

    cv = cross_val_score(model, X_scaled, y, cv=5)
    print(f"[RESULT] CV Accuracy    : {cv.mean():.4f} ± {cv.std():.4f}")

    print("\n" + classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0,
    ))

    plot_confusion_matrix(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        sorted(le.classes_),
        os.path.join(MODEL_DIR, "confusion_matrix.png"),
    )

    for name, obj in [
        ("classifier.pkl",   model),
        ("label_encoder.pkl", le),
        ("scaler.pkl",        scaler),
    ]:
        with open(os.path.join(MODEL_DIR, name), "wb") as f:
            pickle.dump(obj, f)

    print(f"\n[DONE] Model disimpan di '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
