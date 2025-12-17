# =========================================================
# Integrated Pipeline – Updated to Match Test Instructions
# MINIMAL changes over original app.py
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

# =============================
# Load Models
# =============================
partA = PartAModel().to(device)
partA.load_state_dict(torch.load(PARTA_PATH, map_location=device))
partA.eval()

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
# Part B – Known Food
# =============================
def run_partB_known(image_path):
    return predict_image(partB, image_path)

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
def run_partD(image_path, save_path):
    img = preprocess(image_path)
    with torch.no_grad():
        out = torch.sigmoid(partD(img))[0, 0]
    mask = (out > 0.5).cpu().numpy().astype(np.uint8) * 255
    cv2.imwrite(save_path, mask)

# =============================
# Part E – Multi Segmentation
# =============================
def run_partE(image_path, gray_path, color_path):
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

# =============================
# Integrated Test Runner
# =============================
def run_integrated_test(input_dir="Integerated Test", output_dir="Case_I_Output"):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        base = os.path.splitext(img_name)[0]
        out_folder = os.path.join(output_dir, base)
        os.makedirs(out_folder, exist_ok=True)

        label, _ = run_partA(img_path)

        txt_path = os.path.join(out_folder, f"{img_name}_Result.txt")
        with open(txt_path, "w") as f:
            f.write(label + "\n")

            if label == "Fruit":
                fruit_type, _ = run_partC(img_path)
                f.write(fruit_type + "\n")
                f.write("Calories: N/A\n")

                run_partD(img_path, os.path.join(out_folder, "binary_mask.png"))
                run_partE(
                    img_path,
                    os.path.join(out_folder, "multi_mask_gray.png"),
                    os.path.join(out_folder, "multi_mask_color.png"),
                )
            else:
                food_type, _ = run_partB_known(img_path)
                f.write(food_type + "\n")
                f.write("Calories: N/A\n")

# =============================
# Siamese Case II Test
# =============================
def run_siamese_case(anchor_path, reference_dir="Siamese Case II Test"):
    partB.eval()

    # =========================
    # Create output folder
    # =========================
    output_dir = "Case_II_Siamese"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "result.txt")

    anchor_path = os.path.abspath(anchor_path)

    def get_emb(path):
        img = cv2.imread(path)
        if img is None:
            return None

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        x = torch.tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            return partB.forward_one(x).cpu().numpy()[0]

    anchor_emb = get_emb(anchor_path)

    best_img = None
    best_dist = float("inf")

    for ref in os.listdir(reference_dir):
        ref_path = os.path.abspath(os.path.join(reference_dir, ref))

        # Exclude any image whose name starts with "Anchor"
        if ref[:6].lower() == "anchor" or ref[:6].lower() == "Anchor":
            continue


        if not ref.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        ref_emb = get_emb(ref_path)
        if ref_emb is None:
            continue

        dist = np.linalg.norm(anchor_emb - ref_emb)

        if dist < best_dist:
            best_dist = dist
            best_img = ref

    # =========================
    # Save output to TXT
    # =========================
    with open(output_file, "w") as f:
        f.write("Siamese Network - Case II Result\n")
        f.write("--------------------------------\n")
        f.write(f"Anchor image: {os.path.basename(anchor_path)}\n")
        f.write(f"Most similar image: {best_img}\n")
        f.write(f"Distance: {best_dist:.4f}\n")

    print("Result saved in:", output_file)

# =============================
# Main (example usage)
# =============================
if __name__ == "__main__":
    print("If Need (Classification) press 1")
    print("If Need (Siamese Recognition) press 2")

    choice = "1"#input("Enter your choice: ")

    if choice == "1":
        print("Running Part Case I...")
        run_integrated_test()
        print("Case I Finished and Results Saved Successfully")
        
    elif choice == "2":
        print("Running Case II (Siamese Network)...")
        run_siamese_case("Siamese Case II Test/Anchor.jpg")
        print("Case II Finished and Results Saved Successfully")

    else:
        print("Invalid choice! Please enter 1 or 2.")
