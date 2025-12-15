# =========================
# Part E - Fruit Multi-Class Segmentation (Same Structure as Part D)
# =========================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# =========================
# Dataset
# =========================
class FruitMultiClassDataset(Dataset):
    def __init__(self, images, masks, labels):
        self.images = images
        self.masks = masks
        self.labels = labels  # MUST be int (1..30)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        label = int(self.labels[idx])  # 🔥 FORCE INT

        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if isinstance(mask, str):
            mask = cv2.imread(mask, 0)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # background = 0, fruit pixels = label (1..30)
        mask = (mask > 0).astype(np.int64) * label
        mask = torch.from_numpy(mask)

        return img, mask

# =========================
# U-Net Blocks (Same as Part D)
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    def __init__(self, num_classes=31):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.middle = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        mid = self.middle(self.pool(d3))

        u3 = self.up3(mid)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.final(u1)


# =========================
# Pixel Accuracy
# =========================

def pixel_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float().sum()
    return correct / target.numel()


# =========================
# Main Runner
# =========================

def run_partE(fruit, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================================
    # Convert fruit names to numeric labels
    # ======================================
    unique_fruits = sorted(list(set(fruit["train_labels"])))
    fruit_to_idx = {name: i+1 for i, name in enumerate(unique_fruits)}
    # background = 0

    train_labels = [fruit_to_idx[x] for x in fruit["train_labels"]]
    val_labels   = [fruit_to_idx[x] for x in fruit["val_labels"]]

    # =========================
    # Dataset & Loader
    # =========================
    train_ds = FruitMultiClassDataset(
        fruit["train_images"],
        fruit["train_masks"],
        train_labels
    )

    val_ds = FruitMultiClassDataset(
        fruit["val_images"],
        fruit["val_masks"],
        val_labels
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # =========================
    # Model
    # =========================
    model = UNet(num_classes=31).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # =========================
    # Training
    # =========================
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            out = model(imgs)
            loss = criterion(out, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        acc_total = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                acc_total += pixel_accuracy(out, masks).item()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {train_loss/len(train_loader):.4f} "
            f"PixelAcc: {acc_total/len(val_loader):.4f}"
        )

    # =========================
    # Test Script
    # =========================
    def partE_test_script(
        image_path,
        save_gray_path="multiclass_mask_gray.png",
        save_color_path="multiclass_mask_color.png"
    ):
        model.eval()

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
            pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)

        cv2.imwrite(save_gray_path, pred)

        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(31, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]

        h, w = pred.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(31):
            color_mask[pred == c] = colors[c]

        cv2.imwrite(save_color_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        return pred, color_mask

    return model, partE_test_script
