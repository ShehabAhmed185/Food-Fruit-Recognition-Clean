# app.py
import torch
import cv2
import numpy as np
import os
from partA import PartAModel   # import model only
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = PartAModel().to(device)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "partA_food_fruit_classifier.pth"
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()
print("Part A model loaded successfully")

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = torch.tensor(img).permute(2, 0, 1)
    img = img.unsqueeze(0).to(device)

    return img

# -----------------------------
# Prediction Function
# -----------------------------
def classify_image(image_path):
    img = preprocess_image(image_path)

    with torch.no_grad():
        prob = model(img).item()

    result = "Fruit " if prob > 0.5 else "Food "
    confidence = prob if prob > 0.5 else 1 - prob

    print("\n==============================")
    print(f"Image: {image_path}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}")
    print("==============================\n")

# -----------------------------
# Tkinter Image Picker
# -----------------------------
def choose_image():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if file_path:
        classify_image(file_path)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    choose_image()
