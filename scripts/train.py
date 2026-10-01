"""
train.py — Perbandingan 3 arsitektur: ResNet-18, ResNet-50, EfficientNet-B0
Strategi: Feature extraction (backbone beku, hanya head dilatih)

Penggunaan:
  python scripts/train.py --model resnet18
  python scripts/train.py --model resnet50
  python scripts/train.py --model efficientnet_b0
"""

import os, time, argparse, csv
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, models, transforms

BATCH   = 16
EPOCHS  = 10
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA    = 'dataset_split'

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def build_model(model_name, num_classes):
    """Load pretrained model, bekukan backbone, ganti head."""
    if model_name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        opt  = torch.optim.Adam(m.fc.parameters(), lr=1e-3)

    elif model_name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        opt  = torch.optim.Adam(m.fc.parameters(), lr=1e-3)

    elif model_name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        opt = torch.optim.Adam(m.classifier[1].parameters(), lr=1e-3)

    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

    return m.to(DEVICE), opt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['resnet18','resnet50','efficientnet_b0'],
                        required=True, help='Arsitektur model yang dipakai')
    args = parser.parse_args()

    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(os.path.join(DATA, 'train'), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA, 'val'),   val_tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    num_classes = len(train_ds.classes)
    print(f"\n[INFO] Model   : {args.model}")
    print(f"[INFO] Kelas   : {train_ds.classes}")
    print(f"[INFO] Train   : {len(train_ds)} foto | Val: {len(val_ds)} foto")
    print(f"[INFO] Device  : {DEVICE}")
    print(f"[INFO] Epochs  : {EPOCHS}\n")

    model, opt = build_model(args.model, num_classes)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = CosineAnnealingLR(opt, T_max=EPOCHS)

    os.makedirs('results', exist_ok=True)
    log_path  = f'results/log_{args.model}.csv'
    best_acc  = 0.0
    t_start   = time.time()
    epoch_90  = None  # epoch pertama akurasi >= 90%

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_acc'])

        for epoch in range(1, EPOCHS + 1):
            # ── Train ──
            model.train()
            running_loss = 0.0
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                opt.step()
                running_loss += loss.item() * imgs.size(0)
            train_loss = running_loss / len(train_ds)

            # ── Val ──
            model.eval()
            correct = 0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    correct += (model(imgs).argmax(1) == labels).sum().item()
            val_acc = correct / len(val_ds) * 100

            scheduler.step()
            writer.writerow([epoch, f'{train_loss:.4f}', f'{val_acc:.2f}'])
            print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc >= 90 and epoch_90 is None:
                epoch_90 = epoch

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'results/best_{args.model}.pth')

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Model        : {args.model}")
    print(f"Best Val Acc : {best_acc:.2f}%")
    print(f"Waktu Latih  : {elapsed/60:.1f} menit")
    print(f"Epoch @ 90%  : {epoch_90 if epoch_90 else 'tidak tercapai'}")
    print(f"Log          : {log_path}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()