"""
latency.py — Ukur latensi inference ResNet-18 vs MobileNetV3-Small
Penggunaan: python latency.py
"""
import time, torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

DEVICE    = torch.device('cpu')   # CPU, sesuai kondisi laptop
N_RUNS    = 100
IMG_SIZE  = 224

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Dummy image (simulasi frame kamera)
dummy = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
tensor = tf(dummy).unsqueeze(0).to(DEVICE)

candidates = {
    'ResNet-18':          models.resnet18(weights=None),
    'MobileNetV3-Small':  models.mobilenet_v3_small(weights=None),
}

print(f"{'Model':<22} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'FPS est.':>10}")
print('-' * 65)
for name, model in candidates.items():
    model.eval().to(DEVICE)
    times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = model(tensor)
            times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"{name:<22} {avg:>10.2f} {min(times):>10.2f} {max(times):>10.2f} {1000/avg:>10.1f}")
