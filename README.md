# BISINDO Hand Sign Recognition

Project ini adalah sistem pengenalan bahasa isyarat BISINDO berbasis computer vision menggunakan MediaPipe dan machine learning. Aplikasi ini mampu menangkap pose tangan dari kamera, mengekstrak landmark, lalu memprediksi kelas isyarat secara real-time.

## Fitur Utama

- Mengumpulkan data isyarat dari webcam
- Melatih model classifier untuk mengenali simbol/kelas isyarat
- Menjalankan inferensi real-time dari kamera
- Menyediakan antarmuka web sederhana untuk demo

## Struktur Proyek

- collect_data.py: script untuk merekam data landmark tangan
- train_model.py: script pelatihan model dan penyimpanan model
- inference.py: deteksi isyarat secara real-time dari webcam
- inference_kata.py: script inferensi tambahan (jika tersedia)
- web/: berisi file demo web sederhana
- model/: folder hasil training model (akan dibuat saat pelatihan)

## Persyaratan

Pastikan Python sudah terinstall. Kemudian install dependency berikut:

```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib seaborn
```

## Cara Penggunaan

1. Kumpulkan data isyarat

```bash
python collect_data.py
```

Ikuti instruksi di terminal, pilih kelas isyarat, lalu tekan tombol spasi untuk mulai merekam.

2. Latih model

```bash
python train_model.py
```

Model akan disimpan ke folder model/.

3. Jalankan inferensi real-time

```bash
python inference.py
```

Aplikasi akan membuka kamera dan menampilkan hasil prediksi isyarat.

## Output

- Data hasil rekaman: data/landmarks.json
- Model hasil training: model/classifier.pkl
- Label encoder: model/label_encoder.pkl
- Scaler: model/scaler.pkl
- Confusion matrix: model/confusion_matrix.png

## Catatan

- Proyek ini cocok untuk pengembangan dan eksperimen pembelajaran komputer vision.
- Untuk hasil yang lebih baik, gunakan dataset yang lebih banyak dan bervariasi.
- File model dan data besar biasanya tidak disarankan untuk di-commit ke GitHub secara langsung.

## Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan pengembangan mandiri.
