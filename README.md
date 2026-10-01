# AIoT Smart Locker — Face ID Module
**RET503 · Pertemuan 3 · CDIO Stage #2**

## Deskripsi Proyek
Sistem autentikasi wajah untuk smart locker berbasis AIoT. Kamera mendeteksi dan mengklasifikasikan wajah pengguna terdaftar secara real-time. Modul ini bekerja paralel dengan sensor fingerprint fisik.

---

## Spesifikasi Kamera
| Parameter | Nilai |
|---|---|
| Resolusi RGB | 1920 × 1080 @ 30 FPS (MJPG) |
| Resolusi IR | 352 × 352 @ 15 FPS (pengembangan lanjutan) |
| FOV | 95° (lebar — wajib crop ROI) |
| Focus | Fixed-focus — optimal 50–100 cm |
| Windows Hello | 4.x Support |
| Daya | 5V; 900mA (USB) |

> **Penting:** Karena FOV 95°, posisikan wajah **50–80 cm** dari kamera. Wajah terdeteksi otomatis lalu di-crop sebelum resize ke 224×224.

---

## Pipeline Preprocessing (harus identik saat training & deployment)
```
Frame 1920×1080 (BGR)
    → cv2.cvtColor(BGR2RGB)
    → Haar Cascade deteksi wajah
    → Crop ROI + margin 25%
    → Resize 224×224
    → Normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

---

## Struktur Folder
```
smart_locker_faceid/
├── dataset_raw/
│   ├── metadata.csv        ← log semua foto (wajib diisi)
│   ├── ali/
│   ├── budi/
│   ├── ...                 ← satu folder per orang (13 orang)
│   └── unknown/            ← wajah tidak terdaftar
├── dataset_split/
│   ├── train/              ← sesi1 (otomatis oleh split.py)
│   └── val/                ← sesi2+ (otomatis oleh split.py)
├── scripts/
│   ├── capture.py          ← registrasi kamera 1920×1080, crop otomatis
│   ├── split.py            ← pisah train/val berdasarkan sesi
│   ├── train.py            ← latih 3 mode (feature/partial/scratch)
│   └── latency.py          ← ukur latensi inference
└── README.md
```

---

## Cara Penggunaan

### Install
```bash
pip install torch torchvision opencv-python pillow
```

### 1. Ambil Data (Sesi 1 → train)
```bash
# Posisikan wajah 50–80 cm dari kamera!
python scripts/capture.py ali terang --sesi sesi1 --target 50
python scripts/capture.py budi terang --sesi sesi1 --target 50
# ... ulangi untuk semua 13 orang + unknown

# Kontrol:
# SPASI → simpan (crop wajah otomatis)
# S     → simpan tanpa crop (fallback)
# Q     → keluar
```

### 2. Ambil Data (Sesi 2 → val, kondisi berbeda)
```bash
python scripts/capture.py ali redup --sesi sesi2 --target 20
```

### 3. Split Dataset
```bash
python scripts/split.py
# Sesi1 → dataset_split/train/
# Sesi2+ → dataset_split/val/
```

### 4. Latih 3 Mode
```bash
python scripts/train.py --mode feature   # hanya fc, LR=1e-3
python scripts/train.py --mode partial   # layer4+fc, discriminative LR
python scripts/train.py --mode scratch   # semua layer, LR=1e-3
```

### 5. Ukur Latensi
```bash
python scripts/latency.py   # ResNet-18 vs MobileNetV3-Small
```

---

## Hasil Eksperimen

| Mode | Akurasi Val Terbaik | Waktu Latih | Epoch @ 90% | Latensi Inf. |
|------|-------------------|-------------|-------------|-------------|
| feature | _isi setelah praktikum_ | | | |
| partial | | | | |
| scratch | | | | |

*(Grafik: `results/accuracy_plot.png`)*

---

## Analisis

### Hipotesis (sebelum eksperimen)
Feature extraction dan partial fine-tuning diperkirakan jauh lebih baik dari scratch karena dataset kecil (≥50 citra/kelas dari 13 orang). Model pretrained ImageNet sudah belajar fitur tepi, tekstur, dan pola yang relevan untuk pengenalan wajah.

### Hasil dan Temuan
*(Isi setelah praktikum)*

### Model yang Dipilih
*(Isi setelah eksperimen — pertimbangkan akurasi val vs latensi di CPU)*

### Risiko yang Ditemukan
*(Catat data leakage, overfitting, blur akibat fixed-focus, atau isu lain)*

---

## Catatan Penting
- **Anti data leakage**: split berdasarkan SESI, bukan random frame.
- **BGR→RGB**: OpenCV baca BGR, model ImageNet butuh RGB — konversi wajib.
- **Crop wajah dulu**: jangan langsung resize full frame karena FOV 95° membuat wajah kecil.
- **Jarak 50–80 cm**: sesuai fixed-focus kamera, tandai garis di lantai saat sesi capture.
- **metadata.csv**: wajib diisi tiap sesi untuk analisis kegagalan minggu 8 & 12.

---

*Program Studi Teknologi Rekayasa Robotika · Politeknik Negeri Batam*
