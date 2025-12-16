import tkinter as tk
from tkinter import filedialog
import torch
import cv2
import numpy as np
import os
import pickle

# =============================
# Imports from project parts
# =============================
from partA import PartAModel
from partD import UNet as BinaryUNet
from partE import UNet as MultiUNet
from partB import load_partB, predict_image
from partC import build_model

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
PARTB_PREFIX = os.path.join(BASE_DIR, "partB")  # will load partB_model.pth & partB_refs.pkl
PARTC_PATH = os.path.join(BASE_DIR, "partC_classification_model.pth")
PARTD_PATH = os.path.join(BASE_DIR, "unet_partD.pth")
PARTE_PATH = os.path.join(BASE_DIR, "unet_partE.pth")

# =============================
# Load Part A – Food / Fruit
# =============================
partA = PartAModel().to(device)
partA.load_state_dict(torch.load(PARTA_PATH, map_location=device))
partA.eval()

# =============================
# Load Part B – Food Recognition (Siamese)
# =============================
partB = load_partB(PARTB_PREFIX)
partB = partB.to(device)
partB.eval()

# =============================
# Load Part C – Fruit Classification
# =============================
checkpoint_C = torch.load(PARTC_PATH, map_location=device)
num_classes_C = len(checkpoint_C['id2label'])
partC = build_model(num_classes=num_classes_C).to(device)
partC.load_state_dict(checkpoint_C['model'])
partC.eval()
partC_id2label = checkpoint_C['id2label']

# =============================
# Load Part D – Binary Segmentation
# =============================
partD = BinaryUNet().to(device)
partD.load_state_dict(torch.load(PARTD_PATH, map_location=device))
partD.eval()

# =============================
# Load Part E – Multi-class Segmentation
# =============================
partE = MultiUNet(num_classes=31).to(device)
partE.load_state_dict(torch.load(PARTE_PATH, map_location=device))
partE.eval()

print("All models loaded successfully")

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
# Part B – Food Recognition
# =============================
def run_partB(image_path):
    label, conf = predict_image(partB, image_path)
    return label, conf

# =============================
# Part C – Fruit Classification
# =============================
def run_partC(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Invalid image path")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = partC(img_tensor)
        prob = torch.softmax(out, dim=1)
        cid = int(out.argmax(1))
        conf = float(prob[0][cid])

    return partC_id2label[cid], conf

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
def run_partE(image_path,
              gray_path="multi_mask_gray.png",
              color_path="multi_mask_color.png"):

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
# Full Pipeline
# =============================
def run_pipeline(image_path):
    print("\nProcessing:", image_path)

    label, conf = run_partA(image_path)
    print(f"Part A: {label} ({conf:.2f})")

    results = {
        "classification": label,
        "confidence": conf
    }

    if label == "Fruit":
        print("Running Part C, D & E...")

        fruit_type, fruit_conf = run_partC(image_path)
        print(f"Part C: {fruit_type} ({fruit_conf:.2f})")

        results["fruit_type"] = fruit_type
        results["fruit_type_confidence"] = fruit_conf
        results["binary_mask"] = run_partD(image_path)

        gray, color = run_partE(image_path)
        results["multi_mask_gray"] = gray
        results["multi_mask_color"] = color

    else:
        print("Running Part B – Food Recognition")

        food_type, food_conf = run_partB(image_path)
        print(f"Part B: {food_type} ({food_conf:.2f})")

        results["food_type"] = food_type
        results["food_confidence"] = food_conf

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
