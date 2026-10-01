"""
split.py — Pisahkan dataset secara random 80% train / 20% val
Penggunaan: python scripts/split.py
"""
import os, shutil, random

SRC   = 'dataset_raw'
TRAIN = os.path.join('dataset_split', 'train')
VAL   = os.path.join('dataset_split', 'val')
RATIO = 0.8  # 80% train, 20% val
SEED  = 42

random.seed(SEED)

def main():
    # Ambil semua kelas (folder di dataset_raw, kecuali file)
    kelas_list = [k for k in os.listdir(SRC)
                  if os.path.isdir(os.path.join(SRC, k)) and k != 'unknown']

    if not kelas_list:
        print("[ERROR] Tidak ada folder kelas di dataset_raw!")
        return

    print(f"[INFO] Kelas ditemukan: {kelas_list}")
    print(f"[INFO] Split: {int(RATIO*100)}% train / {int((1-RATIO)*100)}% val\n")

    total_train = 0
    total_val   = 0

    for kelas in kelas_list:
        src_folder = os.path.join(SRC, kelas)
        files = [f for f in os.listdir(src_folder) if f.endswith('.png')]
        random.shuffle(files)

        n_train = int(len(files) * RATIO)
        train_files = files[:n_train]
        val_files   = files[n_train:]

        # Buat folder tujuan
        train_dir = os.path.join(TRAIN, kelas)
        val_dir   = os.path.join(VAL,   kelas)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir,   exist_ok=True)

        # Copy file
        for f in train_files:
            shutil.copy2(os.path.join(src_folder, f), os.path.join(train_dir, f))
        for f in val_files:
            shutil.copy2(os.path.join(src_folder, f), os.path.join(val_dir, f))

        print(f"  {kelas}: {len(train_files)} train | {len(val_files)} val")
        total_train += len(train_files)
        total_val   += len(val_files)

    print(f"\n[SELESAI] Total: {total_train} train | {total_val} val")
    print(f"[INFO] Tersimpan di 'dataset_split/train' dan 'dataset_split/val'")

if __name__ == '__main__':
    main()