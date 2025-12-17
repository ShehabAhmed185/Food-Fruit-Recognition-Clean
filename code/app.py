# =========================================================
# Improved & Clean Integration Pipeline (Project-Ready)
# Covers Part A, B (Known + Unseen), C, D, E
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
PARTB_PREFIX = os.path.join(BASE_DIR, "partB")
PARTC_PATH = os.path.join(BASE_DIR, "partC_classification_model.pth")
PARTD_PATH = os.path.join(BASE_DIR, "unet_partD.pth")
PARTE_PATH = os.path.join(BASE_DIR, "unet_partE.pth")

UNSEEN_DIR = os.path.join(BASE_DIR, "unseen")  # folder with unseen food images
USE_UNSEEN_MODE = False  # set True only during unseen-food testing

# =============================
# Load Models
# =============================
partA = PartAModel().to(device)
partA.load_state_dict(torch.load(PARTA_PATH, map_location=device))
partA.eval()

partB = load_partB(PARTB_PREFIX)
partB = load_partB(PARTB_PREFIX, device=device)
partB.eval()

checkpoint_C = torch.load(PARTC_PATH, map_location=device)
num_classes_C = len(checkpoint_C['id2label'])
partC = build_model(num_classes=num_classes_C).to(device)
partC.load_state_dict(checkpoint_C['model'])
partC.eval()
partC_id2label = checkpoint_C['id2label']

partD = BinaryUNet().to(device)
partD.load_state_dict(torch.load(PARTD_PATH, map_location=device))
partD.eval()

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
# Part B – Known Food Recognition
# =============================
def run_partB_known(image_path):
    return predict_image(partB, image_path)

# =============================
# Part B – Unseen Food Recognition
# =============================
def run_partB_unseen(anchor_path, candidate_paths, threshold=0.5):
    def get_embedding(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Invalid image path: {image_path}")

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))

        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = partB.forward_one(x).cpu().numpy()[0]
        return emb

    # Anchor embedding
    anchor_emb = get_embedding(anchor_path)

    best_match = None
    best_dist = 1e9

    # Compare with N unseen images
    for path in candidate_paths:
        emb = get_embedding(path)
        dist = np.linalg.norm(anchor_emb - emb)

        if dist < best_dist:
            best_dist = dist
            best_match = path

    confidence = 1.0 / (1.0 + best_dist)

    if confidence < threshold:
        return "No Match", confidence

    return os.path.basename(best_match), confidence

# =============================
# Part C – Fruit Classification
# =============================
def run_partC(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))

    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = partC(x)
        prob = torch.softmax(out, dim=1)
        cid = int(out.argmax(1))

    return partC_id2label[cid], float(prob[0][cid])

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

    results = {"stage1": label, "confidence": conf}

    if label == "Fruit":
        fruit_type, fruit_conf = run_partC(image_path)
        print(f"Part C => {fruit_type} ({fruit_conf:.2f})")

        results.update({
            "type": fruit_type,
            "type_confidence": fruit_conf,
            "binary_mask": run_partD(image_path),
        })

        gray, color = run_partE(image_path)
        results["multi_mask_gray"] = gray
        results["multi_mask_color"] = color

    else:
        if USE_UNSEEN_MODE:
            candidate_images = [
            os.path.join("Project Data", "Food", "Validation", "ceviche", "217909.jpg"),
            os.path.join("Project Data", "Food", "Validation", "ceviche", "1532642.jpg"),
            os.path.join("Project Data", "Food", "Validation", "ceviche", "2031866.jpg"),
            os.path.join("Project Data", "Food", "Validation", "ceviche", "1315781.jpg"),
            os.path.join("Project Data", "Food", "Validation", "ceviche", "2783027.jpg"),
            os.path.join("Project Data", "Food", "Validation", "ceviche", "3556970.jpg")
            ]

            food_type, food_conf = run_partB_unseen(
                anchor_path=image_path,
                candidate_paths=candidate_images
            )

            print(f"Part B (Unseen): {food_type} ({food_conf:.2f})")
        else:
            food_type, food_conf = run_partB_known(image_path)
            print(f"Part B => {food_type} ({food_conf:.2f})")

        results.update({
            "type": food_type,
            "type_confidence": food_conf
        })

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
