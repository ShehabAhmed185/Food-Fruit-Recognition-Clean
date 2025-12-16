# ================================
# Colab-ready Food Recognition (Part B) - EXACT MATCH WITH partB.py
# ================================

import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import cv2
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# Set ROOT_PATH for Colab
# -------------------------------
ROOT_PATH = "/content/drive/MyDrive/Project_Data/project_data"

# ================================
# Dataset Loading (same logic, Colab-friendly)
# ================================

def load_calories(file_path):
    calories = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if ":" in line:
                    name, cal = line.split(":", 1)
                    if "~" in cal:
                        calories[name.strip()] = float(cal.split("~")[1].split()[0])
    except:
        pass
    return calories

def load_dataset(root_path):
    food_train_images, food_train_labels = [], []
    food_val_images, food_val_labels = [], []

    for split, img_list, lbl_list in [("Train", food_train_images, food_train_labels), ("Validation", food_val_images, food_val_labels)]:
        base = os.path.join(root_path, "Food", split)
        for cat in os.listdir(base):
            cat_path = os.path.join(base, cat)
            if not os.path.isdir(cat_path): continue
            for img in glob(os.path.join(cat_path, "*")):
                img_list.append(img)
                lbl_list.append(cat)

    return {
        "food_train_images": food_train_images,
        "food_train_labels": food_train_labels,
        "food_val_images": food_val_images,
        "food_val_labels": food_val_labels,
        "food_train_cal": load_calories(os.path.join(root_path, "Food", "Train Calories.txt")),
        "food_val_cal": load_calories(os.path.join(root_path, "Food", "Val Calories.txt")),
    }

# ================================
# Image Reading
# ================================

def read_images(paths, size=(224,224)):
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.resize(img, size)
        imgs.append(img)
    return np.array(imgs)

# ================================
# Siamese Dataset (IDENTICAL)
# ================================

class SiamesePairsDataset(Dataset):
    def __init__(self, images, labels, transform=None, negatives_per_sample=1):
        self.images = images.astype('float32')
        self.labels = labels
        self.transform = transform
        self.negatives_per_sample = negatives_per_sample
        self.label_to_indices = {l: np.where(labels == l)[0] for l in np.unique(labels)}
        self.pairs = self._prepare_pairs()

    def _prepare_pairs(self):
        pairs = []
        for i in range(len(self.images)):
            label = self.labels[i]
            pos = [j for j in self.label_to_indices[label] if j != i]
            if pos:
                pairs.append((i, random.choice(pos), 1))
            for _ in range(self.negatives_per_sample):
                neg_label = random.choice([l for l in self.label_to_indices if l != label])
                neg_idx = random.choice(self.label_to_indices[neg_label])
                pairs.append((i, neg_idx, 0))
        random.shuffle(pairs)
        return pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        i1,i2,t = self.pairs[idx]
        return self.images[i1], self.images[i2], torch.tensor(t, dtype=torch.float32)

# ================================
# Siamese Network (EXACT COPY)
# ================================

class SiameseNetwork(nn.Module):
    def __init__(self, input_shape=(224,224,3)):
        super().__init__()
        self.input_shape = input_shape
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg16.features.children()))
        for p in self.feature_extractor.parameters(): p.requires_grad=False
        for p in self.feature_extractor[24:].parameters(): p.requires_grad=True
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512*7*7, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.embedding = nn.Linear(256, 128)
        self.distance_fn = nn.PairwiseDistance(p=2)
        self.reference_embeddings = {}

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.embedding(x)

    def forward(self, x1, x2):
        return self.distance_fn(self.forward_one(x1), self.forward_one(x2))

    @staticmethod
    def contrastive_loss(margin=1.0):
        def loss(out, tgt):
            return torch.mean(tgt*out**2 + (1-tgt)*torch.clamp(margin-out, min=0)**2)
        return loss

    def _preprocess_images_for_vgg(self, imgs):
        imgs = torch.from_numpy(imgs).float()
        imgs = imgs[..., [2,1,0]].permute(0,3,1,2)
        norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        return norm(imgs/255.0)

    def train_model(self, food_data, epochs=10, batch_size=16):
        imgs = self._preprocess_images_for_vgg(food_data['train_images']).numpy()
        labels = np.array(food_data['train_labels'])
        split = int(len(imgs)*0.8)
        train_ds = SiamesePairsDataset(imgs[:split], labels[:split])
        val_ds   = SiamesePairsDataset(imgs[split:], labels[split:])
        train_dl = DataLoader(train_ds, batch_size, True)
        val_dl   = DataLoader(val_ds, batch_size)
        opt = optim.Adam(self.parameters(), 1e-4)
        loss_fn = self.contrastive_loss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        for e in range(epochs):
            self.train()
            for a,b,t in train_dl:
                a,b,t = a.to(device),b.to(device),t.to(device)
                opt.zero_grad()
                loss = loss_fn(self(a,b), t)
                loss.backward(); opt.step()
            print(f"Epoch {e+1}/{epochs} done")

    def save_model(self, path='partB_siamese_model'):
        torch.save(self.state_dict(), path+'.pth')
        with open(path+'_embeddings.pkl','wb') as f:
            pickle.dump(self.reference_embeddings, f)

# ================================

# MAIN
# ================================
if __name__ == '__main__':
    data = load_dataset(ROOT_PATH)
    food = {
        'train_images': read_images(data['food_train_images']),
        'train_labels': data['food_train_labels'],
        'val_images': read_images(data['food_val_images']),
        'val_labels': data['food_val_labels'],
        'train_cal': data['food_train_cal'],
        'val_cal': data['food_val_cal'],
    }
    model = SiameseNetwork()
    model.train_model(food, epochs=1, batch_size=2)
    model.save_model()
    print('Model trained and saved successfully')
