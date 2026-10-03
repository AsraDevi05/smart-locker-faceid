# AIoT Smart Locker — Face ID Module
**RET503 · Pertemuan 3 · CDIO Stage #2 · Analisis Hasil**

---

## Deskripsi Proyek
Sistem autentikasi wajah untuk smart locker berbasis AIoT menggunakan Transfer Learning. Tiga arsitektur CNN pretrained (ImageNet) dibandingkan untuk klasifikasi wajah pengguna terdaftar secara real-time melalui kamera Windows Hello USB (1920×1080, FOV 95°).

**Kelas:** Asra, Ester (2 kelas)
**Dataset:** 50 foto/kelas → 80 train / 20 val (split 80:20 random)
**Strategi TL:** Feature extraction — backbone dibekukan, hanya head yang dilatih

---

## Hasil Eksperimen

### Tabel Perbandingan 3 Arsitektur

| Model | Param (juta) | Best Val Acc | Waktu Latih | Epoch @ 90% | Latensi CPU |
|---|---|---|---|---|---|
| ResNet-18 | ≈11.7 | **95.00%** | 1.0 menit | 4 | 59.92 ms (16.7 FPS) |
| ResNet-50 | ≈25.6 | **100.00%** | 2.2 menit | 3 | - |
| EfficientNet-B0 | ≈5.3 | **100.00%** | 1.5 menit | 2 | - |

### Latensi Inference (CPU, 100 runs)

| Model | Avg (ms) | Min (ms) | Max (ms) | FPS Est. |
|---|---|---|---|---|
| ResNet-18 | 59.92 | 47.30 | 138.50 | 16.7 |
| MobileNetV3-Small | 20.39 | 15.04 | 68.14 | 49.0 |

*(Grafik akurasi per epoch: lihat `results/accuracy_plot.png`)*

---

## Analisis

### Hipotesis Awal
Feature extraction diperkirakan efektif karena dataset kecil (50 foto/kelas) dan domain mirip (foto wajah RGB). Model pretrained ImageNet sudah belajar fitur tepi, tekstur, dan pola wajah yang relevan.

### Temuan Utama

**1. Semua model mencapai akurasi tinggi dengan data kecil**
Ketiga model berhasil mencapai ≥ 90% hanya dengan 40 foto training per kelas. Ini membuktikan bahwa transfer learning sangat efektif untuk dataset kecil — fitur ImageNet yang sudah dipelajari cukup kuat untuk diadaptasi ke domain wajah.

**2. EfficientNet-B0 paling efisien**
EfficientNet-B0 mencapai 100% val accuracy di epoch ke-2, paling cepat di antara ketiganya. Dengan ukuran model hanya ≈5.3 juta parameter (paling kecil), ini menunjukkan desain arsitekturnya yang lebih efisien dibanding ResNet.

**3. ResNet-50 akurat tapi tidak stabil di awal**
ResNet-50 sempat drop ke 60% di epoch 4 setelah mencapai 100% di epoch 3. Ini menunjukkan bahwa dengan data yang sangat kecil, model yang lebih besar justru lebih sulit distabilkan — perlu learning rate schedule yang lebih hati-hati.

**4. ResNet-18 paling stabil**
Meskipun akurasi terbaik hanya 95%, kurva training ResNet-18 paling konsisten dan tidak ada drop drastis. Untuk deployment di edge device, stabilitas ini penting.

**5. Latensi: MobileNetV3-Small jauh lebih cepat**
MobileNetV3-Small (20.39 ms, 49 FPS) hampir 3× lebih cepat dari ResNet-18 (59.92 ms, 16.7 FPS) di CPU. Untuk smart locker yang butuh respons real-time, MobileNetV3-Small lebih cocok meskipun tidak diuji akurasinya di eksperimen ini.

### Model yang Dipilih untuk Deployment
**EfficientNet-B0** — kombinasi terbaik antara akurasi (100%) dan efisiensi model (≈5.3M parameter, waktu latih 1.5 menit). Untuk latensi deployment nanti perlu diukur ulang setelah Mini PC datang.

### Keterbatasan
- Dataset sangat kecil (50 foto/kelas, 2 kelas) — hasil 100% val acc perlu divalidasi dengan lebih banyak data dan lebih banyak kelas
- Semua foto diambil dalam 1 sesi kondisi yang sama (terang) — belum ada variasi sesi untuk menguji robustness
- Latensi EfficientNet-B0 dan ResNet-50 belum diukur — perlu dijalankan `latency.py` yang diupdate

---

## Cara Reproduksi
```bash
# 1. Ambil data
python scripts/capture.py Asra terang --sesi sesi1 --target 50 --cam 1
python scripts/capture.py Ester terang --sesi sesi1 --target 50 --cam 1

# 2. Split dataset
python scripts/split.py

# 3. Training 3 model
python scripts/train.py --model resnet18
python scripts/train.py --model resnet50
python scripts/train.py --model efficientnet_b0

# 4. Ukur latensi
python scripts/latency.py
```

---

*Program Studi Teknologi Rekayasa Robotika · Politeknik Negeri Batam · RET503*
