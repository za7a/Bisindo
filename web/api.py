"""
api.py — Backend Flask untuk BISINDO Detection
================================================

STRUKTUR MODEL:
    model/
        classifier.pkl   → MLP untuk deteksi huruf & angka (per frame)
        label_encoder.pkl→ LabelEncoder untuk MLP
        scaler.pkl       → StandardScaler untuk normalisasi input MLP

    model_kata/
        lstm_model.h5    → LSTM untuk deteksi kata (sequence frame)
        label_encoder.pkl→ LabelEncoder untuk LSTM

CARA MENJALANKAN:
    pip install flask flask-cors scikit-learn numpy joblib tensorflow
    python api.py

ENDPOINT:
    GET  /health          → status server & semua model
    POST /predict/huruf   → MLP: 1 frame → 1 huruf/angka
    POST /predict/kata    → LSTM: N frame → 1 kata
    GET  /classes/huruf   → daftar kelas MLP
    GET  /classes/kata    → daftar kelas LSTM
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════
# PATH MODEL
# Sesuaikan jika struktur folder berbeda.
# ═══════════════════════════════════════════════════════
PATH = {
    'mlp_classifier':   '../model/classifier.pkl',
    'mlp_scaler':       '../model/scaler.pkl',
    'mlp_encoder':      '../model/label_encoder.pkl',
    'lstm_model':       '../model_kata2/lstm_model.h5',
    'lstm_encoder':     '../model_kata2/label_encoder.pkl',
}

# Objek model — dimuat sekali saat server start
models = {
    'mlp':          None,
    'scaler':       None,
    'mlp_encoder':  None,
    'lstm':         None,
    'lstm_encoder': None,
}

# ═══════════════════════════════════════════════════════
# LOAD SEMUA MODEL
# ═══════════════════════════════════════════════════════
def load_models():
    print("\n[→] Memuat model...\n")

    # -- MLP --------------------------------------------------
    if os.path.exists(PATH['mlp_classifier']):
        models['mlp'] = joblib.load(PATH['mlp_classifier'])
        print(f"  [✓] MLP classifier   : {PATH['mlp_classifier']}")
        print(f"      Kelas            : {list(models['mlp'].classes_)}")
    else:
        print(f"  [!] MLP classifier tidak ditemukan: {PATH['mlp_classifier']}")

    if os.path.exists(PATH['mlp_scaler']):
        models['scaler'] = joblib.load(PATH['mlp_scaler'])
        print(f"  [✓] Scaler           : {PATH['mlp_scaler']}")
    else:
        print(f"  [!] Scaler tidak ditemukan: {PATH['mlp_scaler']}")

    if os.path.exists(PATH['mlp_encoder']):
        models['mlp_encoder'] = joblib.load(PATH['mlp_encoder'])
        print(f"  [✓] MLP label encoder: {PATH['mlp_encoder']}")

    # -- LSTM -------------------------------------------------
    if os.path.exists(PATH['lstm_model']):
        # Import TensorFlow di sini agar server tetap jalan
        # walau TF tidak terinstall (mode MLP-only)
        try:
            from tensorflow.keras.models import load_model
            models['lstm'] = load_model(PATH['lstm_model'])
            print(f"  [✓] LSTM model       : {PATH['lstm_model']}")
            print(f"      Input shape      : {models['lstm'].input_shape}")
        except ImportError:
            print("  [!] TensorFlow tidak terinstall — mode kata tidak tersedia")
        except Exception as e:
            print(f"  [!] Gagal load LSTM: {e}")
    else:
        print(f"  [!] LSTM model tidak ditemukan: {PATH['lstm_model']}")

    if os.path.exists(PATH['lstm_encoder']):
        models['lstm_encoder'] = joblib.load(PATH['lstm_encoder'])
        print(f"  [✓] LSTM label encoder: {PATH['lstm_encoder']}")

    print()


# ═══════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models': {
            'mlp_ready':  models['mlp']  is not None and models['scaler'] is not None,
            'lstm_ready': models['lstm'] is not None,
        }
    })


# ═══════════════════════════════════════════════════════
# GET /classes/huruf  &  /classes/kata
# ═══════════════════════════════════════════════════════
@app.route('/classes/huruf', methods=['GET'])
def classes_huruf():
    if models['mlp'] is None:
        return jsonify({'error': 'MLP belum dimuat'}), 503
    return jsonify({'classes': list(models['mlp'].classes_)})

@app.route('/classes/kata', methods=['GET'])
def classes_kata():
    if models['lstm_encoder'] is None:
        return jsonify({'error': 'LSTM encoder belum dimuat'}), 503
    return jsonify({'classes': list(models['lstm_encoder'].classes_)})


# ═══════════════════════════════════════════════════════
# POST /predict/huruf
#
# Request body:
#   { "landmarks": [x0,y0,z0, ..., x20,y20,z20, x0,y0,z0, ..., x20,y20,z20] }
#   → 126 nilai float (2 tangan × 21 landmark × 3 koordinat)
#   Urutan: [tangan kiri (63), tangan kanan (63)]
#   Jika salah satu tangan tidak terdeteksi, slotnya diisi 0.
#
# Response:
#   { "label": "A", "confidence": 0.97 }
#
# ALUR:
#   landmarks (126) → scaler.transform() → mlp.predict()
#   KENAPA scaler? MLP sensitif terhadap skala nilai.
#   Scaler memastikan input punya distribusi sama seperti saat training.
# ═══════════════════════════════════════════════════════
@app.route('/predict/huruf', methods=['POST'])
def predict_huruf():
    if models['mlp'] is None or models['scaler'] is None:
        return jsonify({'error': 'Model MLP atau scaler belum dimuat'}), 503

    data = request.get_json(force=True)
    if not data or 'landmarks' not in data:
        return jsonify({'error': 'Field "landmarks" tidak ada'}), 400

    landmarks = data['landmarks']

    # 2 tangan × 21 landmark × 3 (x,y,z) = 126
    if len(landmarks) != 126:
        return jsonify({
            'error': f'Panjang landmarks harus 126, diterima: {len(landmarks)}'
        }), 400

    try:
        X = np.array(landmarks, dtype=np.float32).reshape(1, -1)

        # Normalisasi dengan scaler yang sama seperti saat training
        X_scaled = models['scaler'].transform(X)

        # Prediksi → returns integer (karena training pakai le.fit_transform)
        label_int  = models['mlp'].predict(X_scaled)[0]
        proba      = models['mlp'].predict_proba(X_scaled)[0]
        confidence = float(np.max(proba))

        # Decode integer → string label asli via LabelEncoder
        # Dari training_model.py: y = le.fit_transform(y_raw) → integer
        # jadi hasil predict() adalah integer, harus di-inverse_transform
        if models['mlp_encoder'] is not None:
            label = str(models['mlp_encoder'].inverse_transform([label_int])[0])
        else:
            # Fallback: jika tidak ada encoder, coba ambil dari classes_ model
            label = str(label_int)

        return jsonify({
            'label':      label,
            'confidence': round(confidence, 4),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════
# POST /predict/kata
#
# Request body:
#   {
#     "sequence": [
#       [x0,...,x20,y20,z20, x0,...,x20,y20,z20],  ← frame 1 (126 nilai: kiri+kanan)
#       ...                                          ← total N frame
#     ]
#   }
#
# Response:
#   { "label": "halo", "confidence": 0.89 }
#
# ALUR:
#   sequence (N×126) → reshape (1, N, 126) → lstm.predict()
#
# KENAPA reshape 3D?
#   LSTM butuh input (batch, timesteps, features).
#   batch=1, timesteps=N frame, features=126.
# ═══════════════════════════════════════════════════════
@app.route('/predict/kata', methods=['POST'])
def predict_kata():
    if models['lstm'] is None:
        return jsonify({'error': 'Model LSTM belum dimuat'}), 503

    data = request.get_json(force=True)
    if not data or 'sequence' not in data:
        return jsonify({'error': 'Field "sequence" tidak ada'}), 400

    sequence = data['sequence']

    # Validasi: tiap frame harus punya 126 nilai (2 tangan × 63)
    for i, frame in enumerate(sequence):
        if len(frame) != 126:
            return jsonify({
                'error': f'Frame ke-{i} panjangnya {len(frame)}, harus 126'
            }), 400

    try:
        # Reshape ke (1, N_frames, 126) untuk LSTM
        X = np.array(sequence, dtype=np.float32).reshape(1, len(sequence), 126)

        # Prediksi
        proba      = models['lstm'].predict(X, verbose=0)[0]
        class_idx  = int(np.argmax(proba))
        confidence = float(np.max(proba))

        # Decode label
        if models['lstm_encoder'] is not None:
            label = str(models['lstm_encoder'].inverse_transform([class_idx])[0])
        else:
            label = str(class_idx)

        return jsonify({
            'label':      label,
            'confidence': round(confidence, 4),
            'frames_used': len(sequence),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  BISINDO Detection API")
    print("="*50)

    load_models()

    print("[→] Server berjalan di http://localhost:5000")
    print("[→] Tekan Ctrl+C untuk stop\n")

    app.run(host='0.0.0.0', port=5000, debug=True)