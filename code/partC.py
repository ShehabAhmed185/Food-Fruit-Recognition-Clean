# ===========================================================
#                 PART C – Fruit Classification
#      Train/Val Automatic Split – Compatible with main.py
#                 Optimized for MX230 GPU
# ===========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ===========================================================
#                 DATASET CLASS
# ===========================================================
class FruitDataset(Dataset):
    def __init__(self, images, labels, is_train=True):
        self.images = images
        self.labels = labels
        self.is_train = is_train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.is_train:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)

            angle = np.random.uniform(-12, 12)
            M = cv2.getRotationMatrix2D((112,112), angle, 1.0)
            img = cv2.warpAffine(img, M, (224,224), borderMode=cv2.BORDER_REFLECT)

            if np.random.rand() < 0.4:
                alpha = np.random.uniform(0.85, 1.15)
                beta  = np.random.uniform(-15, 15)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2,0,1))

        return torch.tensor(img, dtype=torch.float32), label


# ===========================================================
#                 BUILD MODEL
# ===========================================================
def build_model(num_classes):
    model = models.mobilenet_v3_small(pretrained=False)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, num_classes)
    return model


# ===========================================================
#                 TRAIN / VALIDATE
# ===========================================================
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    preds, trues = [], []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds.extend(out.argmax(1).cpu().tolist())
        trues.extend(labels.cpu().tolist())

    acc = accuracy_score(trues, preds)
    return total_loss / len(loader.dataset), acc


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            out = model(images)
            loss = loss_fn(out, labels)

            total_loss += loss.item() * images.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            trues.extend(labels.cpu().tolist())

    acc = accuracy_score(trues, preds)
    return total_loss / len(loader.dataset), acc


# ===========================================================
#                 MAIN FUNCTION (with splitting)
# ===========================================================
def run_partC(fruit, epochs=10, batch_size=8):

    print("\n====================")
    print(" PART C: Fruit Classification with Split")
    print("====================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------------
    # Merge user train + val into one dataset then split again
    # --------------------------------------------------------
    all_images = np.concatenate([fruit["train_images"], fruit["val_images"]], axis=0)
    all_labels = fruit["train_labels"] + fruit["val_labels"]

    print("Merged dataset size:", len(all_images))

    # --------------------------------------------------------
    # Convert string labels to integers
    # --------------------------------------------------------
    classes = sorted(list(set(all_labels)))
    label2id = {c:i for i,c in enumerate(classes)}
    id2label = {i:c for c,i in label2id.items()}

    numeric_labels = [label2id[c] for c in all_labels]

    # --------------------------------------------------------
    # Train/Val split (80/20)
    # --------------------------------------------------------
    idx = np.arange(len(all_images))

    idx_train, idx_val = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=numeric_labels
    )

    train_images = all_images[idx_train]
    train_labels = [numeric_labels[i] for i in idx_train]

    val_images = all_images[idx_val]
    val_labels = [numeric_labels[i] for i in idx_val]

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------
    train_ds = FruitDataset(train_images, train_labels, is_train=True)
    val_ds   = FruitDataset(val_images, val_labels, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = build_model(num_classes=len(classes)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, loss_fn, device)

        print(f"Train => Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   => Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label
            }, "best_partC.pth")

            print("Saved best model!")

    print("\nTraining Finished!")
    print("Best Val Accuracy =", best_acc)

    # --------------------------------------------------------
    # Test script
    # --------------------------------------------------------
    def test_script(img):
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485,0.456,0.406])
        std  = np.array([0.229,0.224,0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2,0,1))

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(img)
            prob = torch.softmax(out, dim=1)
            cid = int(out.argmax(1))
            conf = float(prob[0][cid])

        return {"category": id2label[cid], "confidence": conf}

    return model, test_script
