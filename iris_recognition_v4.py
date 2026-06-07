# %% [markdown]
# # Iris Recognition — ResNet18 + ArcFace (Production Refactor v4)
#
# **Key changes over v3:**
# 1. Daugman Rubber Sheet Model replaces naive resize — only pure iris tissue is fed to the network
# 2. Input geometry changed from 224×224 to 64×512 (polar iris strip)
# 3. HoughCircles-based pupil/iris boundary localization
# 4. Attention penalty adapted for rectangular feature maps
# 5. All other fixes from v3 retained (subject-exclusive split, ArcFace margin, etc.)

# %% — Cell 1: Install dependencies
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "opencv-python-headless", "scikit-learn", "matplotlib",
                       "onnx", "grad-cam"])

# %% — Cell 2: Imports & Reproducibility
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from math import pi

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

print(f"PyTorch: {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% [markdown]
# ## Configuration

# %% — Cell 3: Configuration
PLATFORM = "local"
if os.path.exists("/kaggle/input"):  PLATFORM = "kaggle"
elif os.path.exists("/content"):     PLATFORM = "colab"
print(f"Platform: {PLATFORM}")

if PLATFORM == "kaggle":
    _root = "/kaggle/input/datasets/sondosaabed/casia-iris-thousand/CASIA-Iris-Thousand/CASIA-Iris-Thousand"
    _save = "/kaggle/working/iris_model_v4"
elif PLATFORM == "colab":
    _root = "/content/dataset/CASIA-Iris-Thousand/CASIA-Iris-Thousand"
    _save = "/content/drive/MyDrive/iris_model_v4"
else:
    _root = "./data/CASIA-Iris-Thousand"
    _save = "./checkpoints_v4"

CONFIG = {
    "dataset_root":    _root,
    "save_dir":        _save,
    "polar_height":    64,
    "polar_width":     512,
    "batch_size":      64,
    "epochs":          40,
    "lr":              1e-3,
    "train_pool_frac": 0.80,
    "val_img_frac":    0.10,
    "patience":        10,
    "min_samples":     3,
    "norm_mean":       0.449,
    "norm_std":        0.226,
    "arcface_s":       64.0,
    "arcface_m":       0.40,
    "attn_weight":     2.0,
}
os.makedirs(CONFIG["save_dir"], exist_ok=True)
assert os.path.isdir(CONFIG["dataset_root"]), f"Dataset not found: {CONFIG['dataset_root']}"
print(f"Dataset : {CONFIG['dataset_root']}")
print(f"Save dir: {CONFIG['save_dir']}")

# %% [markdown]
# ## Preprocessing: Daugman Rubber Sheet Model
#
# Replaces the naive letterbox-resize with proper iris segmentation and polar unrolling:
# 1. **Pupil localization** via `cv2.HoughCircles` on blurred/thresholded image
# 2. **Iris localization** via `cv2.HoughCircles` on edge-detected image, constrained by pupil
# 3. **Rubber Sheet mapping**: for each (θ, r) in polar space, compute Cartesian coordinates
#    using Daugman's model and sample with bilinear interpolation

# %% — Cell 4: Daugman Rubber Sheet Preprocessing
def find_pupil_circle(gray):
    """Detect the pupillary boundary using HoughCircles."""
    h, w = gray.shape
    min_dim = min(h, w)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold to enhance dark pupil region
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # HoughCircles on the thresholded image
    circles = cv2.HoughCircles(
        thresh,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dim // 4,
        param1=100,
        param2=30,
        minRadius=max(10, min_dim // 20),
        maxRadius=min_dim // 4,
    )
    if circles is None:
        return None

    # Pick the circle closest to the image center
    circles = circles[0]
    cx_img, cy_img = w / 2, h / 2
    best = min(circles, key=lambda c: (c[0] - cx_img)**2 + (c[1] - cy_img)**2)
    return best  # (x, y, r)


def find_iris_circle(gray, pupil_xyr):
    """Detect the limbus (iris outer) boundary, constrained by the pupil."""
    if pupil_xyr is None:
        return None

    h, w = gray.shape
    min_dim = min(h, w)
    px, py, pr = pupil_xyr

    # Use Canny edges for the iris boundary
    edges = cv2.Canny(gray, 30, 100)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dim // 3,
        param1=50,
        param2=30,
        minRadius=int(pr * 1.5),
        maxRadius=min(int(min_dim * 0.45), w // 2),
    )
    if circles is None:
        return None

    # Pick the circle whose center is closest to the pupil center
    # and whose radius is larger than the pupil's
    circles = circles[0]
    valid = [c for c in circles if c[2] > pr * 1.2]
    if not valid:
        return None

    best = min(valid, key=lambda c: (c[0] - px)**2 + (c[1] - py)**2)
    return best  # (x, y, r)


def daugman_rubber_sheet(gray, pupil_xyr, iris_xyr, polar_h=64, polar_w=512):
    """
    Daugman's Rubber Sheet Model — unroll the iris annulus into a rectangular
    polar representation using bilinear interpolation.

    For each angle θ ∈ [0, 2π) and radial step r ∈ (0, 1]:
        x(r, θ) = (1-r) * pupil_x(θ) + r * iris_x(θ)
        y(r, θ) = (1-r) * pupil_y(θ) + r * iris_y(θ)
    """
    h, w = gray.shape
    px, py, pr = pupil_xyr
    ix, iy, ir = iris_xyr

    # Angle array: polar_w points around the circle
    theta = np.linspace(0, 2 * pi, polar_w, endpoint=False)

    # Pupil boundary points
    px_circle = px + pr * np.cos(theta)
    py_circle = py + pr * np.sin(theta)

    # Iris boundary points
    ix_circle = ix + ir * np.cos(theta)
    iy_circle = iy + ir * np.sin(theta)

    # Radial steps: polar_h points from pupil to iris
    radius = np.linspace(1 / polar_h, 1.0, polar_h).reshape(-1, 1)  # (polar_h, 1)

    # Compute Cartesian coordinates for each (r, θ)
    x_coords = (1 - radius) * px_circle + radius * ix_circle  # (polar_h, polar_w)
    y_coords = (1 - radius) * py_circle + radius * iy_circle  # (polar_h, polar_w)

    # Bilinear interpolation via cv2.remap
    map_x = x_coords.astype(np.float32)
    map_y = y_coords.astype(np.float32)

    polar_img = cv2.remap(
        gray, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return polar_img.astype(np.uint8)


# Track segmentation statistics
_seg_stats = {"total": 0, "success": 0, "fail": 0}

def preprocess_iris_to_polar(image_path, polar_h=64, polar_w=512):
    """
    Full preprocessing pipeline: load NIR image → segment pupil & iris → polar unroll.
    Returns a (polar_h, polar_w) uint8 numpy array, or zeros if segmentation fails.
    """
    global _seg_stats
    _seg_stats["total"] += 1

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        _seg_stats["fail"] += 1
        return np.zeros((polar_h, polar_w), dtype=np.uint8)

    # Step 1: Find pupil
    pupil = find_pupil_circle(img)
    if pupil is None:
        _seg_stats["fail"] += 1
        return np.zeros((polar_h, polar_w), dtype=np.uint8)

    # Step 2: Find iris
    iris = find_iris_circle(img, pupil)
    if iris is None:
        _seg_stats["fail"] += 1
        return np.zeros((polar_h, polar_w), dtype=np.uint8)

    # Step 3: Quality check — pupil-to-iris ratio
    alpha = pupil[2] / iris[2]
    if alpha < 0.1 or alpha > 0.8:
        _seg_stats["fail"] += 1
        return np.zeros((polar_h, polar_w), dtype=np.uint8)

    # Step 4: Rubber sheet mapping
    polar = daugman_rubber_sheet(img, pupil, iris, polar_h, polar_w)
    _seg_stats["success"] += 1
    return polar

print("Daugman preprocessing defined")
print(f"Polar output: {CONFIG['polar_height']}×{CONFIG['polar_width']}")

# %% [markdown]
# ## Dataset Class

# %% — Cell 5: Dataset Class
class IrisDataset(Dataset):
    """
    Preloads polar iris images into RAM. Applies normalization consistently.
    Augmentations mimic realistic edge-device degradations.
    """
    def __init__(self, image_paths, labels,
                 polar_h=64, polar_w=512, augment=False,
                 mean=0.449, std=0.226):
        self.labels  = labels
        self.polar_h = polar_h
        self.polar_w = polar_w
        self.augment  = augment
        self.mean     = mean
        self.std      = std

        print(f"Preloading {len(image_paths)} images (polar {polar_h}×{polar_w})...")
        self.cache = [preprocess_iris_to_polar(p, polar_h, polar_w) for p in image_paths]
        print(f"Preload complete — seg success: {_seg_stats['success']}, fail: {_seg_stats['fail']}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.cache[idx].copy()
        label = self.labels[idx]

        if self.augment:
            img = self._augment(img)

        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor, label

    def _augment(self, img):
        """Augmentations for polar iris strips."""
        h, w = img.shape

        # Circular shift (simulates iris torsion / rotation in polar space)
        if random.random() > 0.3:
            shift = random.randint(-w // 8, w // 8)
            img = np.roll(img, shift, axis=1)

        # Contrast / brightness shift
        alpha = random.uniform(0.75, 1.25)
        beta  = random.randint(-30, 30)
        img   = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Gaussian blur
        if random.random() > 0.6:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Horizontal band occlusion (simulates eyelid in polar space)
        if random.random() > 0.5:
            band_h = random.randint(h // 10, h // 4)
            if random.random() > 0.5:
                img[:band_h, :] = 0  # top
            else:
                img[-band_h:, :] = 0  # bottom

        # JPEG compression artifact
        if random.random() > 0.6:
            quality = random.randint(40, 90)
            _, buf  = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img     = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

        return img

print("IrisDataset defined")

# %% [markdown]
# ## Subject-Exclusive Data Split

# %% — Cell 6: Subject-Exclusive Data Split
from sklearn.model_selection import train_test_split as sk_split

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def discover_casia(root, min_samples=3):
    all_paths, all_labels, label_names, subject_map = [], [], [], {}
    label_idx = 0

    for subj in sorted(os.listdir(root)):
        subj_dir = os.path.join(root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for eye in ["L", "R"]:
            eye_dir = os.path.join(subj_dir, eye)
            if not os.path.isdir(eye_dir):
                continue
            images = [os.path.join(eye_dir, f)
                      for f in os.listdir(eye_dir)
                      if os.path.splitext(f)[1].lower() in EXTS]
            if len(images) >= min_samples:
                all_paths.extend(images)
                all_labels.extend([label_idx] * len(images))
                label_names.append(f"{subj}_{eye}")
                subject_map[label_idx] = subj
                label_idx += 1

    return all_paths, all_labels, label_names, subject_map

all_paths, all_labels, label_names, subject_map = discover_casia(
    CONFIG["dataset_root"], CONFIG["min_samples"]
)
NUM_CLASSES_TOTAL = len(label_names)
print(f"Total classes (subject+eye): {NUM_CLASSES_TOTAL}")
print(f"Total images: {len(all_paths)}")

# %% — Cell 7: Split execution
unique_subjects = sorted({subject_map[i] for i in range(NUM_CLASSES_TOTAL)})
random.seed(SEED)
random.shuffle(unique_subjects)

n_total      = len(unique_subjects)
n_train_pool = int(CONFIG["train_pool_frac"] * n_total)

train_subjects = set(unique_subjects[:n_train_pool])
test_subjects  = set(unique_subjects[n_train_pool:])

assert train_subjects.isdisjoint(test_subjects), "Subject overlap detected!"
print(f"Train-pool subjects : {len(train_subjects)}")
print(f"Held-out test subjects: {len(test_subjects)}")

train_class_set   = {i for i in range(NUM_CLASSES_TOTAL) if subject_map[i] in train_subjects}
train_class_remap = {old: new for new, old in enumerate(sorted(train_class_set))}
NUM_CLASSES       = len(train_class_set)

pool_paths  = [p for p, l in zip(all_paths, all_labels) if subject_map[l] in train_subjects]
pool_labels = [train_class_remap[l]
               for p, l in zip(all_paths, all_labels)
               if subject_map[l] in train_subjects]

train_paths, val_paths, train_labels, val_labels = sk_split(
    pool_paths, pool_labels,
    test_size=CONFIG["val_img_frac"],
    stratify=pool_labels,
    random_state=SEED,
)

test_paths_raw  = [p for p, l in zip(all_paths, all_labels) if subject_map[l] in test_subjects]
test_labels_raw = [l for p, l in zip(all_paths, all_labels) if subject_map[l] in test_subjects]

print(f"\nNUM_CLASSES (ArcFace head) : {NUM_CLASSES}")
print(f"Train images : {len(train_paths):,}  |  Val images : {len(val_paths):,}")
print(f"Test  images : {len(test_paths_raw):,}")

# %% — Cell 8: DataLoader creation
# Reset segmentation stats before preloading
_seg_stats = {"total": 0, "success": 0, "fail": 0}

train_ds = IrisDataset(train_paths, train_labels,
                       polar_h=CONFIG["polar_height"], polar_w=CONFIG["polar_width"],
                       augment=True, mean=CONFIG["norm_mean"], std=CONFIG["norm_std"])
val_ds   = IrisDataset(val_paths, val_labels,
                       polar_h=CONFIG["polar_height"], polar_w=CONFIG["polar_width"],
                       augment=False, mean=CONFIG["norm_mean"], std=CONFIG["norm_std"])
test_ds  = IrisDataset(test_paths_raw, test_labels_raw,
                       polar_h=CONFIG["polar_height"], polar_w=CONFIG["polar_width"],
                       augment=False, mean=CONFIG["norm_mean"], std=CONFIG["norm_std"])

print(f"\nSegmentation stats — Total: {_seg_stats['total']}, "
      f"Success: {_seg_stats['success']}, Fail: {_seg_stats['fail']}")
if _seg_stats['total'] > 0:
    print(f"Success rate: {_seg_stats['success']/_seg_stats['total']*100:.1f}%")

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=2, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)
print("DataLoaders ready")

# %% [markdown]
# ## Model: IrisResNet18 + ArcFace
#
# ResNet18 adapted for 1-channel 64×512 polar iris input.
# The adaptive average pooling handles the non-square geometry naturally.
# Feature map sizes through the network:
# - conv1 (stride 2): 32×256
# - maxpool (stride 2): 16×128
# - layer1: 16×128, layer2: 8×64, layer3: 4×32, layer4: 2×16

# %% — Cell 9: Model Architecture
class IrisResNet(nn.Module):
    """
    ResNet18 for 1-channel polar iris images (64×512).
    Returns (embedding, layer3_spatial, layer4_spatial).
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights  = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # 1-channel conv1: average across RGB dim of pretrained weights
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3   # 256 ch, 4×32 for 64×512 input
        self.layer4  = backbone.layer4   # 512 ch, 2×16 for 64×512 input
        self.avgpool = backbone.avgpool
        self.embedding_dim = 512

        if freeze_backbone:
            for m in [self.conv1, self.bn1, self.layer1, self.layer2]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        l3_feat = x
        x = self.layer4(x)
        l4_feat = x
        x = self.avgpool(x)
        x = x.flatten(1)
        embeds = F.normalize(x, p=2, dim=1)
        return embeds, l3_feat, l4_feat

    def get_embedding(self, x):
        embeds, _, _ = self.forward(x)
        return embeds


class ArcFaceHead(nn.Module):
    """Additive Angular Margin (ArcFace) classification head."""
    def __init__(self, embedding_dim, num_classes, s=64.0, m=0.50):
        super().__init__()
        self.s   = s
        self.m   = m
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        W      = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, W)
        theta  = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        output  = cosine * (1 - one_hot) + target * one_hot
        return output * self.s

    def get_cosine(self, embeddings):
        W = F.normalize(self.weight, p=2, dim=1)
        return F.linear(embeddings, W)


model       = IrisResNet(pretrained=True).to(DEVICE)
arcface     = ArcFaceHead(512, NUM_CLASSES,
                          s=CONFIG["arcface_s"], m=CONFIG["arcface_m"]).to(DEVICE)
base_model  = model

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    base_model = model.module

n_backbone = sum(p.numel() for p in base_model.parameters())
n_head     = sum(p.numel() for p in arcface.parameters())
print(f"IrisResNet18: {n_backbone:,} params")
print(f"ArcFace head: {n_head:,} params  (s={arcface.s}, m={arcface.m}, classes={NUM_CLASSES})")

# Verify feature map sizes
with torch.no_grad():
    _dummy = torch.randn(1, 1, CONFIG["polar_height"], CONFIG["polar_width"]).to(DEVICE)
    _, _l3, _l4 = base_model(_dummy)
    print(f"Layer3 feature map: {_l3.shape}")
    print(f"Layer4 feature map: {_l4.shape}")

# %% [markdown]
# ## Attention Penalty (Spatial Regularization)
#
# Adapted for the rectangular layer3 feature map (4×32 for 64×512 input).
# Uses separate sigma values for height and width dimensions.

# %% — Cell 10: Attention Penalty (Removed)
# Spatial regularization is no longer needed since Daugman Rubber Sheet model
# fundamentally isolates the pure iris tissue, removing the background.

# %% [markdown]
# ## Training Loop

# %% — Cell 11: Training Loop
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(
    [{"params": model.parameters()},
     {"params": arcface.parameters()}],
    lr=CONFIG["lr"], weight_decay=1e-2
)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")
patience_ctr  = 0

for epoch in range(CONFIG["epochs"]):
    # ── Train ──
    model.train(); arcface.train()
    run_loss, correct, total = 0.0, 0, 0
    attn_w = CONFIG["attn_weight"] * min(1.0, epoch / 10)

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()

        embeds, _, _ = model(imgs)
        logits   = arcface(embeds, lbls)
        cls_loss = criterion(logits, lbls)
        loss = cls_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        run_loss += loss.item() * imgs.size(0)
        with torch.no_grad():
            cosine = arcface.get_cosine(embeds)
        correct += (cosine.argmax(1) == lbls).sum().item()
        total   += lbls.size(0)

    train_loss = run_loss / total
    train_acc  = correct  / total

    # ── Validation ──
    model.eval(); arcface.eval()
    vl_loss, vl_correct, vl_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            embeds, _, _ = model(imgs)
            cosine = arcface.get_cosine(embeds)
            loss   = criterion(cosine, lbls)
            vl_loss    += loss.item() * imgs.size(0)
            vl_correct += (cosine.argmax(1) == lbls).sum().item()
            vl_total   += lbls.size(0)

    val_loss = vl_loss    / vl_total
    val_acc  = vl_correct / vl_total

    scheduler.step()
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d}/{CONFIG['epochs']} | "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
          f"AttnW {attn_w:.2f} LR {lr_now:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr  = 0
        torch.save({
            "model_state_dict":   base_model.state_dict(),
            "arcface_state_dict": arcface.state_dict(),
            "num_classes":        NUM_CLASSES,
            "label_names":        label_names,
            "epoch":              epoch,
            "val_acc":            val_acc,
            "config":             CONFIG,
        }, os.path.join(CONFIG["save_dir"], "best_model.pth"))
        print(f"  → Saved best model (val_loss={val_loss:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= CONFIG["patience"]:
            print(f"  → Early stopping at epoch {epoch+1}")
            break

print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")

# %% — Cell 12: Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history["train_loss"], label="Train", lw=2)
ax1.plot(history["val_loss"],   label="Val",   lw=2)
ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss Curves"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history["train_acc"], label="Train", lw=2)
ax2.plot(history["val_acc"],   label="Val",   lw=2)
ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy Curves"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["save_dir"], "training_curves.png"), dpi=150)
plt.show()

# %% [markdown]
# ## Open-Set Evaluation: EER, TAR@FAR, AUC

# %% — Cell 13: Load best checkpoint & evaluate
ckpt = torch.load(os.path.join(CONFIG["save_dir"], "best_model.pth"),
                  map_location=DEVICE, weights_only=False)
base_model.load_state_dict(ckpt["model_state_dict"])
arcface.load_state_dict(ckpt["arcface_state_dict"])
base_model.eval(); arcface.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}")

def extract_embeddings(loader, model, device):
    embeds_list, labels_list = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            emb, _, _ = model(imgs.to(device))
            embeds_list.append(emb.cpu().numpy())
            labels_list.append(lbls.numpy())
    return np.concatenate(embeds_list), np.concatenate(labels_list)

def compute_eer_auc(embeds, labels, n_pairs=100_000, seed=0):
    rng    = np.random.default_rng(seed)
    N      = len(embeds)
    idx_all = np.arange(N)

    genuine_pairs, impostor_pairs = [], []
    unique_lbls = np.unique(labels)

    for lbl in unique_lbls:
        idxs = idx_all[labels == lbl]
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                genuine_pairs.append((idxs[i], idxs[j]))

    n_genuine  = len(genuine_pairs)
    if n_genuine == 0:
        imp_a, imp_b = np.array([]), np.array([])
        impostor_pairs = []
    else:
        imp_a      = rng.integers(0, N, size=n_genuine * 3)
        imp_b      = rng.integers(0, N, size=n_genuine * 3)
        imp_mask   = labels[imp_a] != labels[imp_b]
        imp_a, imp_b = imp_a[imp_mask][:n_genuine], imp_b[imp_mask][:n_genuine]
        impostor_pairs = list(zip(imp_a, imp_b))

    if len(genuine_pairs) == 0 or len(impostor_pairs) == 0:
        print("WARNING: Not enough pairs generated for evaluation. Check segmentation failure rates.")
        return {
            "eer": 50.0,
            "auc": 0.5,
            "tar_at_01far": 0.0,
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "scores": np.array([]),
            "is_genuine": np.array([])
        }

    all_pairs = genuine_pairs + impostor_pairs
    is_genuine = np.array([1] * len(genuine_pairs) + [0] * len(impostor_pairs))
    a_idx = np.array([p[0] for p in all_pairs])
    b_idx = np.array([p[1] for p in all_pairs])
    scores = (embeds[a_idx] * embeds[b_idx]).sum(axis=1)

    fpr, tpr, thr = roc_curve(is_genuine, scores)
    roc_auc = auc(fpr, tpr)

    fnr     = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100

    far_target = 0.001
    tar_at_far = tpr[np.searchsorted(fpr, far_target, side="right") - 1]

    return {"eer": eer, "auc": roc_auc, "tar_at_01far": tar_at_far,
            "fpr": fpr, "tpr": tpr, "scores": scores, "is_genuine": is_genuine}

print("Extracting test embeddings...")
test_embeds, test_labels_arr = extract_embeddings(test_loader, base_model, DEVICE)
print(f"Embeddings: {test_embeds.shape}")

metrics = compute_eer_auc(test_embeds, test_labels_arr)
print(f"\n=== Open-Set Evaluation (Unseen Subjects) ===")
print(f"EER:              {metrics['eer']:.3f}%")
print(f"AUC:              {metrics['auc']:.4f}")
print(f"TAR @ FAR=0.1%%:  {metrics['tar_at_01far']*100:.2f}%")

# %% — Cell 14: ROC & Score Distribution Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(metrics["fpr"], metrics["tpr"], lw=2,
         label=f"AUC={metrics['auc']:.4f}  EER={metrics['eer']:.2f}%")
ax1.plot([0, 1], [0, 1], "--", color="gray")
ax1.set(xlabel="FAR", ylabel="TAR", title="ROC Curve — Open-Set Test Subjects")
ax1.legend(); ax1.grid(True, alpha=0.3)

gen_scores = metrics["scores"][metrics["is_genuine"] == 1]
imp_scores = metrics["scores"][metrics["is_genuine"] == 0]
bins = np.linspace(-0.2, 1.0, 80)
ax2.hist(imp_scores, bins=bins, alpha=0.6, label="Impostor", color="red",   density=True)
ax2.hist(gen_scores, bins=bins, alpha=0.6, label="Genuine",  color="green", density=True)
ax2.set(xlabel="Cosine Similarity", ylabel="Density",
        title="Score Distribution — Genuine vs Impostor")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["save_dir"], "openset_evaluation.png"), dpi=150)
plt.show()

# %% [markdown]
# ## t-SNE Visualization

# %% — Cell 15: t-SNE
# Use a subset of test embeddings for t-SNE
N_TSNE = min(2000, len(test_embeds))
tsne_idx = np.random.choice(len(test_embeds), N_TSNE, replace=False)
tsne_emb = test_embeds[tsne_idx]
tsne_lbl = test_labels_arr[tsne_idx]

print(f"Running t-SNE on {N_TSNE} embeddings...")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
tsne_2d = tsne.fit_transform(tsne_emb)

unique_tsne_labels = np.unique(tsne_lbl)
n_colors = min(len(unique_tsne_labels), 20)

fig, ax = plt.subplots(figsize=(10, 10))
for i, lbl in enumerate(unique_tsne_labels[:n_colors]):
    mask = tsne_lbl == lbl
    ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], s=10, alpha=0.7, label=f"ID {lbl}")
if n_colors <= 20:
    ax.legend(markerscale=3, fontsize=7, loc="best")
ax.set_title("t-SNE — Test Embeddings (Unseen Subjects)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["save_dir"], "tsne.png"), dpi=150)
plt.show()

# %% [markdown]
# ## GradCAM++ Visualization

# %% — Cell 16: GradCAM++
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class GradCAMWrapper(nn.Module):
    """Wrapper that returns class logits (cosine) for GradCAM targeting."""
    def __init__(self, backbone, arcface_head):
        super().__init__()
        self.backbone = backbone
        self.arcface_head = arcface_head
    def forward(self, x):
        emb, _, _ = self.backbone(x)
        return self.arcface_head.get_cosine(emb)

wrapped = GradCAMWrapper(base_model, arcface).eval()

# Use layer3 for GradCAM (4×32 for 64×512 input — good spatial resolution)
target_layer = base_model.layer3[-1]
cam = GradCAMPlusPlus(model=wrapped, target_layers=[target_layer])

N_VIS = min(6, len(val_ds))
fig, axes = plt.subplots(3, N_VIS, figsize=(3 * N_VIS, 9))
if N_VIS == 1:
    axes = axes.reshape(3, 1)

for i in range(N_VIS):
    t, lbl = val_ds[i]
    inp = t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = wrapped(inp).argmax(1).item()

    gcam = cam(inp, targets=[ClassifierOutputTarget(pred)])[0]

    # Original polar strip
    orig = t[0].numpy()
    orig_disp = ((orig * CONFIG["norm_std"] + CONFIG["norm_mean"]) * 255).clip(0, 255).astype(np.uint8)

    # GradCAM heatmap
    heatmap_color = cv2.applyColorMap((gcam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match polar strip dimensions
    heatmap_resized = cv2.resize(heatmap_color, (CONFIG["polar_width"], CONFIG["polar_height"]))
    orig_3ch = cv2.cvtColor(orig_disp, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(orig_3ch, 0.5, heatmap_resized, 0.5, 0)

    axes[0, i].imshow(orig_disp, cmap="gray", aspect="auto")
    axes[0, i].set_title(f"Polar Strip (lbl={lbl})", fontsize=8); axes[0, i].axis("off")

    axes[1, i].imshow(heatmap_resized, aspect="auto")
    axes[1, i].set_title("GradCAM++", fontsize=8); axes[1, i].axis("off")

    axes[2, i].imshow(overlay, aspect="auto")
    axes[2, i].set_title("Overlay", fontsize=8); axes[2, i].axis("off")

plt.suptitle(f"GradCAM++ — Layer3 targeting cosine class logits (polar {CONFIG['polar_height']}×{CONFIG['polar_width']})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["save_dir"], "gradcam.png"), dpi=150)
plt.show()

# %% [markdown]
# ## ONNX Export

# %% — Cell 17: ONNX Export
class ONNXWrapper(nn.Module):
    """Export-safe wrapper — tracer sees only the 1-D embedding output."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        emb, _, _ = self.backbone(x)
        return emb

export_model = ONNXWrapper(base_model).eval()
dummy        = torch.randn(1, 1, CONFIG["polar_height"], CONFIG["polar_width"]).to(DEVICE)
onnx_path    = os.path.join(CONFIG["save_dir"], "iris_resnet18_v4.onnx")

torch.onnx.export(
    export_model,
    dummy,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
    input_names=["iris_polar"],
    output_names=["embedding"],
    dynamic_axes={
        "iris_polar": {0: "batch_size"},
        "embedding":  {0: "batch_size"},
    },
)
print(f"ONNX saved: {onnx_path}  ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

import onnx
onnx.checker.check_model(onnx.load(onnx_path))
print("ONNX graph verified")

# %% [markdown]
# ## Unit Tests

# %% — Cell 18: Unit Tests
import traceback

PASS = "  PASS"
FAIL = "  FAIL"

def run_test(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
    except Exception as e:
        print(f"{FAIL}  {name}")
        traceback.print_exc()

def test_no_subject_overlap():
    assert train_subjects.isdisjoint(test_subjects)
    assert all(0 <= l < NUM_CLASSES for l in val_labels)

def test_polar_output_shape():
    """Verify that preprocessing outputs correct polar shape."""
    if len(val_ds.cache) > 0:
        assert val_ds.cache[0].shape == (CONFIG["polar_height"], CONFIG["polar_width"])

def test_normalization_consistency():
    tensor, _ = val_ds[0]
    mean_val = tensor.mean().item()
    assert abs(mean_val) < 1.5, f"Tensor mean {mean_val:.3f}"

def test_arcface_logit_ordering():
    arcface.eval()
    dummy_emb = F.normalize(torch.randn(4, 512).to(DEVICE), dim=1)
    dummy_lbl = torch.zeros(4, dtype=torch.long).to(DEVICE)
    arcface_logits = arcface(dummy_emb, dummy_lbl)
    cosine_logits  = arcface.get_cosine(dummy_emb)
    gt_arcface = arcface_logits[:, 0] / arcface.s
    gt_cosine  = cosine_logits[:, 0]
    assert (gt_arcface <= gt_cosine + 1e-4).all().item()

def test_embedding_is_unit_norm():
    base_model.eval()
    dummy = torch.randn(4, 1, CONFIG["polar_height"], CONFIG["polar_width"]).to(DEVICE)
    with torch.no_grad():
        emb, _, _ = base_model(dummy)
    norms = torch.norm(emb, dim=1)
    assert (norms - 1.0).abs().max().item() < 1e-5

# test_attention_penalty_scalar removed because attention_penalty is no longer used

def test_gradcam_output_shape():
    dummy = torch.randn(1, 1, CONFIG["polar_height"], CONFIG["polar_width"]).to(DEVICE)
    with torch.no_grad():
        out = wrapped(dummy)
    assert out.shape == (1, NUM_CLASSES)

def test_onnx_batch1():
    try:
        import onnxruntime as ort
    except ImportError:
        print("    (onnxruntime not installed → skipping)")
        return
    sess  = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 1, CONFIG["polar_height"], CONFIG["polar_width"]).astype(np.float32)
    out   = sess.run(None, {"iris_polar": dummy})[0]
    assert out.shape == (1, 512)
    norm  = np.linalg.norm(out[0])
    assert abs(norm - 1.0) < 1e-5

def test_eer_sane():
    assert 0 <= metrics["eer"] <= 50
    assert 0.5 <= metrics["auc"] <= 1.0

print("Running unit tests...\n")
run_test("T1:  No subject overlap / val labels valid",    test_no_subject_overlap)
run_test("T2:  Polar output shape correct",               test_polar_output_shape)
run_test("T3:  Tensor normalization in valid range",       test_normalization_consistency)
run_test("T4:  ArcFace margin suppresses GT logit",        test_arcface_logit_ordering)
run_test("T5:  Embeddings are L2-normalized",              test_embedding_is_unit_norm)
run_test("T7:  GradCAM wrapper returns class logits",      test_gradcam_output_shape)
run_test("T8:  ONNX batch=1 inference succeeds",           test_onnx_batch1)
run_test("T9:  EER in sane range (<50%)",                  test_eer_sane)

# %% [markdown]
# ## Done
#
# Artifacts saved to `CONFIG['save_dir']`:
# - `best_model.pth` — PyTorch checkpoint
# - `iris_resnet18_v4.onnx` — ONNX model (dynamic batch, opset 17)
# - `training_curves.png`, `openset_evaluation.png`, `tsne.png`, `gradcam.png`
#
# **Next step (Jetson Nano):**
# ```bash
# trtexec --onnx=iris_resnet18_v4.onnx \
#         --saveEngine=iris_resnet18.trt \
#         --fp16 \
#         --minShapes=iris_polar:1x1x64x512 \
#         --optShapes=iris_polar:1x1x64x512 \
#         --maxShapes=iris_polar:4x1x64x512
# ```
