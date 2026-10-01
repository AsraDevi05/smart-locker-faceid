"""
capture.py — Registrasi wajah langsung dari kamera (YuNet)
Kamera: Windows Hello USB, Fixed-focus, FOV 95°
YuNet: deteksi wajah modern, tahan masker, wajah miring, kondisi ramai

Penggunaan:
  python scripts/capture.py Asra terang --sesi sesi1 --target 50 --cam 1

Kontrol:
  SPASI  → simpan foto (crop wajah terbesar otomatis)
  S      → simpan tanpa crop (fallback)
  Q      → keluar
"""

import cv2, os, csv, argparse
from datetime import date

MARGIN      = 0.30
MODEL_PATH  = 'face_detection_yunet.onnx'

def load_yunet(w, h):
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH, "", (w, h),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5
    )
    return detector

def detect_faces(detector, frame_rgb, min_size=80):
    """Deteksi wajah pakai YuNet, filter yang terlalu kecil."""
    _, faces = detector.detect(frame_rgb)
    if faces is None:
        return []
    # Filter wajah kecil (orang jauh di background)
    return [f for f in faces if f[2] >= min_size and f[3] >= min_size]

def crop_largest(frame_rgb, faces, margin=MARGIN):
    """Crop wajah terbesar (terdekat ke kamera)."""
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3])
    H, W = frame_rgb.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx);  y1 = max(0, y - my)
    x2 = min(W, x + w + mx); y2 = min(H, y + h + my)
    return frame_rgb[y1:y2, x1:x2], (x1, y1, x2, y2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kelas',  help='Nama orang / kelas')
    parser.add_argument('cahaya', help='terang / redup / backlight')
    parser.add_argument('--sesi',    default='sesi1')
    parser.add_argument('--jarak',   default='sedang')
    parser.add_argument('--target',  type=int, default=50)
    parser.add_argument('--cam',     type=int, default=0)
    parser.add_argument('--minface', type=int, default=80,
                        help='Ukuran minimum wajah dalam piksel (default 80)')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model YuNet tidak ditemukan: {MODEL_PATH}")
        print("[INFO]  Jalankan dulu:")
        print('  curl -L "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -o face_detection_yunet.onnx')
        return

    folder = os.path.join('dataset_raw', args.kelas)
    os.makedirs(folder, exist_ok=True)

    existing  = [f for f in os.listdir(folder) if f.endswith('.png')]
    counter   = len(existing) + 1
    tanggal   = date.today().strftime('%Y%m%d')
    meta_path = os.path.join('dataset_raw', 'metadata.csv')
    write_header = not os.path.exists(meta_path)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = load_yunet(W, H)

    print(f"\n[INFO] Kelas: '{args.kelas}' | Sesi: {args.sesi} | Target: {args.target}")
    print(f"[INFO] YuNet aktif — tahan masker & wajah miring")
    print(f"[TIP]  Posisikan wajah 50–70 cm dari kamera")
    print(f"\n       SPASI=foto (crop otomatis)  |  S=simpan tanpa crop  |  Q=keluar\n")

    saved = 0
    with open(meta_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(['nama_file','kelas','tanggal','sesi',
                             'kondisi_cahaya','jarak','ekspresi','catatan'])

        while saved < args.target:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # YuNet butuh RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces     = detect_faces(detector, frame_rgb, min_size=args.minface)

            preview  = frame_bgr.copy()
            best_box = None

            if len(faces) > 0:
                # Gambar semua kotak hijau
                for f in faces:
                    x,y,w,h = int(f[0]),int(f[1]),int(f[2]),int(f[3])
                    cv2.rectangle(preview, (x,y), (x+w,y+h), (0,255,0), 2)
                # Kotak biru = wajah terbesar
                _, (x1,y1,x2,y2) = crop_largest(frame_rgb, faces)
                cv2.rectangle(preview, (x1,y1), (x2,y2), (255,100,0), 3)
                cv2.putText(preview, "BIRU=disimpan", (x1, max(y1-8,20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,0), 2)
                best_box = (x1,y1,x2,y2)

            status = f"Wajah: {len(faces)}  |  {saved}/{args.target}  |  SPASI=foto  S=no-crop  Q=keluar"
            cv2.putText(preview, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(preview, f"{args.kelas} | {args.sesi} | {args.cahaya}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow('Registrasi Wajah — Smart Locker', cv2.resize(preview, (960,540)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Keluar.")
                break

            elif key in (ord(' '), ord('s')):
                if key == ord(' '):
                    if best_box is None:
                        print("  [!] Wajah tidak terdeteksi. Hadap kamera atau tekan S.")
                        continue
                    x1,y1,x2,y2 = best_box
                    crop = frame_rgb[y1:y2, x1:x2]
                    catatan = 'yunet_crop'
                else:
                    crop = frame_rgb
                    catatan = 'no_crop'

                nama_file = f"{args.kelas}_{tanggal}_{args.sesi}_{args.cahaya}_{counter:03d}.png"
                path = os.path.join(folder, nama_file)
                cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                writer.writerow([nama_file, args.kelas, tanggal, args.sesi,
                                 args.cahaya, args.jarak, 'netral', catatan])
                print(f"  [+] {nama_file} ({crop.shape[1]}×{crop.shape[0]})")
                saved  += 1
                counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SELESAI] {saved} foto tersimpan di '{folder}'")

if __name__ == '__main__':
    main()