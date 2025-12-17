# ================================
# Part B - Improved & Clean Siamese Network (Project-Ready)
# ================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import os
import cv2
import random
import pickle
import matplotlib.pyplot as plt
import warnings
import csv
import time

warnings.filterwarnings("ignore", category=UserWarning)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(
        224, scale=(0.75, 1.0), ratio=(0.9, 1.1)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# =========================================================
# Dataset: On-the-fly Siamese pairs (memory safe)
# =========================================================
class SiamesePairsDataset(Dataset):
    def __init__(self, images, labels, transform=None, negatives_per_sample=1):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.negatives_per_sample = negatives_per_sample

        self.label_to_indices = {
            l: np.where(labels == l)[0] for l in np.unique(labels)
        }
        self.pairs = self._build_pairs()

    def _build_pairs(self):
        pairs = []
        for i, label in enumerate(self.labels):
            pos_candidates = self.label_to_indices[label]
            pos_candidates = pos_candidates[pos_candidates != i]

            if len(pos_candidates) > 0:
                j = np.random.choice(pos_candidates)
                pairs.append((i, j, 1))

            neg_labels = [l for l in self.label_to_indices if l != label]
            for _ in range(self.negatives_per_sample):
                nl = np.random.choice(neg_labels)
                j = np.random.choice(self.label_to_indices[nl])
                pairs.append((i, j, 0))

        random.shuffle(pairs)
        return pairs

    def __getitem__(self, idx):
        i, j, y = self.pairs[idx]

        img1 = self.images[i]
        img2 = self.images[j]

        # BGR → RGB (FIX)
        img1 = img1[..., ::-1].copy()
        img2 = img2[..., ::-1].copy()


        # IMPORTANT: same random seed → same transform
        seed = random.randint(0, 99999)
        random.seed(seed)
        torch.manual_seed(seed)

        if self.transform:
            img1 = self.transform(img1)
            random.seed(seed)
            torch.manual_seed(seed)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

# =========================================================
# Siamese Network
# =========================================================
class SiameseNetwork(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()
        self.input_size = input_size

        # -------- Encoder (VGG16) --------
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.encoder = base.features

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.encoder[24:].parameters():
            p.requires_grad = True

        # -------- Projection Head --------
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        self.distance = nn.PairwiseDistance(p=2)
        self.reference_embeddings = {}

    def forward_one(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.distance(self.forward_one(x1), self.forward_one(x2))


# =========================================================
# Loss & Metrics
# =========================================================
def contrastive_loss(margin=1.0):
    def loss_fn(d, y):
        pos = y * d.pow(2)
        neg = (1 - y) * torch.clamp(margin - d, min=0).pow(2)
        return (pos + neg).mean()
    return loss_fn


def distance_accuracy(th=0.5):
    def acc_fn(d, y):
        return ((d < th).float() == y).float().mean().item()
    return acc_fn


def init_partB_logger():
    os.makedirs("partB_logs", exist_ok=True)
    log_path = "partB_logs/training_log.csv"

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "epoch_time_sec"
            ])
    return log_path


# =========================================================
# Utilities
# =========================================================
def preprocess(images):
    """
    images: numpy array (N, H, W, 3) BGR from OpenCV
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()

    # BGR -> RGB (SAFE way)
    images = images[..., [2, 1, 0]]

    # (N, H, W, C) -> (N, C, H, W)
    images = images.permute(0, 3, 1, 2)

    images = images / 255.0

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return normalize(images)


# =========================================================
# Training
# =========================================================
def train_partB(food_data, epochs=10, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    log_path = init_partB_logger()
    X = np.array(food_data['train_images'])
    y = np.array(food_data['train_labels'])

    split = int(0.8 * len(X))
    train_ds = SiamesePairsDataset(
        X[:split], y[:split], transform=train_transform
    )

    val_ds = SiamesePairsDataset(
        X[split:], y[split:], transform=val_transform
    )


    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)

    model = SiameseNetwork().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = contrastive_loss()
    acc_fn = distance_accuracy()

    best_val_acc = 0.0

    for e in range(epochs):
        start_time = time.time()
        model.train()
        tl, ta = 0, 0
        for a, b, yb in train_loader:
            a, b, yb = a.to(device), b.to(device), yb.to(device)
            opt.zero_grad()
            d = model(a, b)
            loss = loss_fn(d, yb)
            loss.backward()
            opt.step()
            tl += loss.item()
            ta += acc_fn(d, yb)

        model.eval()
        vl, va = 0, 0
        with torch.no_grad():
            for a, b, yb in val_loader:
                a, b, yb = a.to(device), b.to(device), yb.to(device)
                d = model(a, b)
                vl += loss_fn(d, yb).item()
                va += acc_fn(d, yb)

        print(f"Epoch {e+1}/{epochs} | "
              f"Loss {tl/len(train_loader):.4f} Acc {ta/len(train_loader):.4f} | "
              f"ValLoss {vl/len(val_loader):.4f} ValAcc {va/len(val_loader):.4f}")



        epoch_time = time.time() - start_time

        train_loss = tl / len(train_loader)
        train_acc  = ta / len(train_loader)
        val_loss   = vl / len(val_loader)
        val_acc    = va / len(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "partB_best_model.pth")


        print(
            f"Epoch {e+1}/{epochs} | "
            f"Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"ValLoss {val_loss:.4f} ValAcc {val_acc:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                e + 1,
                round(train_loss, 6),
                round(train_acc, 6),
                round(val_loss, 6),
                round(val_acc, 6),
                round(epoch_time, 2)
        ])




    return model

# =========================================================
# Reference Embeddings & Prediction
# =========================================================
@torch.no_grad()
def build_reference_embeddings(model, food_data, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    refs = {}

    images = np.array(food_data['train_images'])
    labels = np.array(food_data['train_labels'])

    num_samples = len(images)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        batch_imgs = images[start:end]
        batch_labels = labels[start:end]

        x = preprocess(batch_imgs).to(device)

        emb = model.forward_one(x).cpu().numpy()

        for e, lab in zip(emb, batch_labels):
            if lab not in refs:
                refs[lab] = []
            refs[lab].append(e)

        del x
        torch.cuda.empty_cache()

    model.reference_embeddings = refs
    print(f"Reference embeddings built for {len(refs)} classes")
    return refs

@torch.no_grad()
def predict_image(model, image_path, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")
    img = cv2.resize(img, (224,224))
    x = preprocess(np.expand_dims(img,0)).to(next(model.parameters()).device)

    q = model.forward_one(x).cpu().numpy()[0]
    best_label, best_dist = None, 1e9

    for lab, embs in model.reference_embeddings.items():
        d = np.mean([np.linalg.norm(q-e) for e in embs])
        if d < best_dist:
            best_dist, best_label = d, lab

    conf = 1.0 / (1.0 + best_dist)
    if conf < threshold:
        return 'Unknown', conf
    return best_label, conf


# =========================================================
# Save / Load
# =========================================================
def save_partB(model, path='partB'):
    torch.save(model.state_dict(), path + '_model.pth')
    with open(path + '_refs.pkl', 'wb') as f:
        pickle.dump(model.reference_embeddings, f)


def load_partB(path='partB', device='cpu'):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(path + '_model.pth', map_location=device))
    with open(path + '_refs.pkl', 'rb') as f:
        model.reference_embeddings = pickle.load(f)
    return model



# =========================================================
# Runner (Used by main project)
# =========================================================
def run_partB(food_data, epochs=10, batch_size=16):
    print('='*60)
    print('PART B - Siamese Network Food Recognition')
    print('='*60)

    model = train_partB(food_data, epochs, batch_size)

    model.load_state_dict(torch.load("partB_best_model.pth", map_location="cpu"))

    build_reference_embeddings(model, food_data, batch_size=4)

    save_partB(model)

    return model


