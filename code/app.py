import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import cv2
import numpy as np
import re
import os
from partA import PartAModel
from partD import UNet as BinaryUNet
from partE import UNet as MultiUNet
from partB import SiameseNetwork
from partC import build_model # <--- 1. NEW IMPORT

# =============================
# Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARTA_PATH = os.path.join(BASE_DIR, "partA_food_fruit_classifier.pth")
PARTD_PATH = os.path.join(BASE_DIR, "unet_partD.pth")
PARTE_PATH = os.path.join(BASE_DIR, "unet_partE.pth")
PARTC_PATH = os.path.join(BASE_DIR, "partC_classification_model.pth") # <--- 2. NEW PATH

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

# Load Part C Model (Fruit Classification) # <--- 3. NEW LOAD LOGIC
checkpoint_C = torch.load(PARTC_PATH, map_location=device)
num_classes_C = len(checkpoint_C['id2label'])
partC = build_model(num_classes=num_classes_C).to(device)
partC.load_state_dict(checkpoint_C['model'])
partC.eval()
partC_id2label = checkpoint_C['id2label'] # Store the label map globally

print(" All models loaded successfully")

# =============================
# Preprocessing
# =============================
def preprocess(image_path, resize=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resize:
        img = cv2.resize(img, resize)

    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

# =============================
# Part A – Classification
# =============================
def run_partA(image_path):
    img = preprocess(image_path, resize=(224, 224))
    with torch.no_grad():
        prob = partA(img).item()

    label = "Fruit" if prob > 0.5 else "Food"
    confidence = prob if label == "Fruit" else 1 - prob

    return label, confidence

# =============================
# Part C – Fruit Classification # <--- 4. NEW FUNCTION
# =============================
def run_partC(image_path):
    # Load and preprocess image specifically for the MobileNetV3 model (Part C)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Invalid image path: {image_path}")
    
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224 (as done in partC.py)
    img = cv2.resize(img, (224, 224))
    
    # Normalization (Matching partC.py's test_script)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))

    # Convert to tensor and move to device
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    # Inference
    partC.eval()
    with torch.no_grad():
        out = partC(img_tensor)
        prob = torch.softmax(out, dim=1)
        cid = int(out.argmax(1))
        conf = float(prob[0][cid])
        
    category = partC_id2label[cid]

    return category, conf


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

    # Save gray mask
    cv2.imwrite(gray_path, pred)

    # Color mask
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
# FULL PIPELINE
# =============================
def run_pipeline(image_path):
    print(f"\n Processing: {image_path}")

    label, conf = run_partA(image_path)
    print(f"Part A  {label} ({conf:.2f})")

    results = {
        "classification": label,
        "confidence": conf
    }

    if label == "Fruit":
        print("Running Part C, D & Part E...") # Adjusted print
        
        # New: Run Part C - Fruit Classification
        fruit_type, fruit_conf = run_partC(image_path)
        print(f"Part C  Fruit Type: {fruit_type} ({fruit_conf:.2f})")
        results["fruit_type"] = fruit_type
        results["fruit_type_confidence"] = fruit_conf

        results["binary_mask"] = run_partD(image_path)
        gray, color = run_partE(image_path)
        results["multi_mask_gray"] = gray
        results["multi_mask_color"] = color

        print(" Segmentation completed")

    else:
        # NOTE: You would integrate Part B logic here
        print("Food detected => segmentation skipped (Part B integration required for full pipeline)")

    return results


def choose_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return
    return file_path


# =============================
# Test
# =============================
if __name__ == "__main__":
    test_image = choose_image()
    if test_image:
        output = run_pipeline(test_image)
        print("\nFinal Output:", output)