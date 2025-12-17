# =========================================================
# Clean Integration Pipeline
# TESTING: Part A (Food/Fruit) + Part D (Binary Seg) + Part E (Multi Seg)
# =========================================================

import tkinter as tk
from tkinter import filedialog
import torch
import cv2
import numpy as np
import os

# =============================
# Imports from project parts
# =============================
from partA import PartAModel
from partD import UNet as BinaryUNet
from partE import UNet as MultiUNet

# =============================
# Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARTA_PATH = os.path.join(BASE_DIR, "partA_food_fruit_classifier.pth")
PARTD_PATH = os.path.join(BASE_DIR, "unet_partD.pth")
PARTE_PATH = os.path.join(BASE_DIR, "unet_partE.pth")

# =============================
# Load Models
# =============================
partA = PartAModel().to(device)
partA.load_state_dict(torch.load(PARTA_PATH, map_location=device))
partA.eval()

partD = BinaryUNet().to(device)
partD.load_state_dict(torch.load(PARTD_PATH, map_location=device))
partD.eval()

partE = MultiUNet(num_classes=31).to(device)
partE.load_state_dict(torch.load(PARTE_PATH, map_location=device))
partE.eval()

print("Models A, D, E loaded successfully")

# =============================
# Common preprocessing
# =============================
def preprocess(image_path, resize=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img = cv2.resize(img, resize)

    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

# =============================
# Part A – Food / Fruit
# =============================
def run_partA(image_path):
    img = preprocess(image_path, resize=(224, 224))
    with torch.no_grad():
        prob = partA(img).item()

    label = "Fruit" if prob > 0.5 else "Food"
    confidence = prob if label == "Fruit" else 1 - prob
    return label, confidence

# =============================
# Part D – Binary Segmentation
# =============================
def run_partD(image_path, save_path="binary_mask.png"):
    img = preprocess(image_path)
    with torch.no_grad():
        out = torch.sigmoid(partD(img))[0, 0]

    mask = (out > 0.5).cpu().numpy().astype(np.uint8) * 255
    cv2.imwrite(save_path, mask)
    return save_path

# =============================
# Part E – Multi-class Segmentation
# =============================
def run_partE(image_path, gray_path="multi_gray.png", color_path="multi_color.png"):
    img = preprocess(image_path)
    with torch.no_grad():
        out = partE(img)
        pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)

    cv2.imwrite(gray_path, pred)

    np.random.seed(42)
    colors = np.random.randint(0, 255, (31, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    h, w = pred.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(31):
        color_mask[pred == c] = colors[c]

    cv2.imwrite(color_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    return gray_path, color_path

# =============================
# Integrated Pipeline
# =============================
def run_pipeline(image_path):
    print("\nProcessing:", image_path)

    label, conf = run_partA(image_path)
    print(f"Part A => {label} ({conf:.2f})")

    results = {
        "stage1": label,
        "confidence": conf
    }

    if label == "Fruit":
        results["binary_mask"] = run_partD(image_path)
        gray, color = run_partE(image_path)
        results["multi_mask_gray"] = gray
        results["multi_mask_color"] = color
    else:
        print("Image classified as Food — skipping segmentation")

    return results

# =============================
# UI Helper
# =============================
def choose_image():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

# =============================
# Main
# =============================
if __name__ == "__main__":
    img_path = choose_image()
    if img_path:
        output = run_pipeline(img_path)
        print("\nFinal Output:\n", output)
