# partB.py
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
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# Dataset for on-the-fly pairs
# -------------------------------
class SiamesePairsDataset(Dataset):
    """Generate Siamese pairs on-the-fly to save memory."""
    def __init__(self, images, labels, transform=None, negatives_per_sample=1):
        self.images = images.astype('float32')
        self.labels = labels
        self.transform = transform
        self.negatives_per_sample = negatives_per_sample
        self.label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        self.pairs = self._prepare_pairs()

    def _prepare_pairs(self):
        pairs = []
        for idx in range(len(self.images)):
            current_label = self.labels[idx]
            pos_indices = [i for i in self.label_to_indices[current_label] if i != idx]
            if pos_indices:
                pos_idx = np.random.choice(pos_indices)
                pairs.append((idx, pos_idx, 1))
            neg_labels = [l for l in self.label_to_indices.keys() if l != current_label]
            for _ in range(self.negatives_per_sample):
                if not neg_labels:
                    continue
                neg_label = np.random.choice(neg_labels)
                neg_idx = np.random.choice(self.label_to_indices[neg_label])
                pairs.append((idx, neg_idx, 0))
        np.random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, target = self.pairs[idx]
        img1, img2 = self.images[idx1], self.images[idx2]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(target, dtype=torch.float32)

# -------------------------------
# Siamese Network
# -------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self, input_shape=(224, 224, 3)):
        super(SiameseNetwork, self).__init__()
        self.input_shape = input_shape
        self.encoder = self.build_encoder()
        self.distance_fn = nn.PairwiseDistance(p=2)
        self.reference_embeddings = {}
        self.label_encoder = LabelEncoder()

    def build_encoder(self):
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        encoder = nn.Sequential(*list(vgg16.features.children()))
        for param in encoder.parameters():
            param.requires_grad = False
        for param in encoder[24:].parameters():
            param.requires_grad = True
        self.feature_extractor = encoder
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.embedding = nn.Linear(256, 128)
        return self

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.embedding(x)
        return x

    def forward(self, input1, input2):
        embedding1 = self.forward_one(input1)
        embedding2 = self.forward_one(input2)
        distance = self.distance_fn(embedding1, embedding2)
        return distance

    @staticmethod
    def contrastive_loss(margin=1.0):
        def loss(output, target):
            positive_loss = target * torch.pow(output, 2)
            negative_loss = (1 - target) * torch.pow(torch.clamp(margin - output, min=0.0), 2)
            return torch.mean(positive_loss + negative_loss)
        return loss

    @staticmethod
    def distance_accuracy(threshold=0.5):
        def acc(output, target):
            pred_same = (output < threshold).float()
            return (pred_same == target).float().mean().item()
        return acc

    def _preprocess_images_for_vgg(self, images):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        if images.shape[-1] == 3:
            images = images[..., [2,1,0]]  # BGR→RGB
            images = images.permute(0,3,1,2)
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        return normalize(images / 255.0)

    # -------------------------------
    # Training using on-the-fly pairs
    # -------------------------------
    def train_model(self, food_data, epochs=20, batch_size=16, validation_split=0.2):
        print("Preparing training data...")
        images = np.array(food_data['train_images']).astype('float32')
        labels = np.array(food_data['train_labels'])
        preprocessed = self._preprocess_images_for_vgg(images).numpy()

        split_idx = int(len(images) * (1 - validation_split))
        train_images, train_labels = preprocessed[:split_idx], labels[:split_idx]
        val_images, val_labels = preprocessed[split_idx:], labels[split_idx:]

        transform = transforms.Compose([])  # Already normalized
        train_dataset = SiamesePairsDataset(train_images, train_labels, transform=transform)
        val_dataset = SiamesePairsDataset(val_images, val_labels, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        criterion = self.contrastive_loss()
        accuracy_metric = self.distance_accuracy()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"Using device: {device}")

        history = {'loss': [], 'distance_accuracy': [], 'val_loss': [], 'val_distance_accuracy': []}

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            train_acc = 0.0
            for batch1, batch2, batch_labels in train_loader:
                batch1, batch2, batch_labels = batch1.to(device), batch2.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                distances = self(batch1, batch2)
                loss = criterion(distances, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += accuracy_metric(distances, batch_labels)
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_acc / len(train_loader)

            # Validation
            self.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for batch1, batch2, batch_labels in val_loader:
                    batch1, batch2, batch_labels = batch1.to(device), batch2.to(device), batch_labels.to(device)
                    distances = self(batch1, batch2)
                    loss = criterion(distances, batch_labels)
                    val_loss += loss.item()
                    val_acc += accuracy_metric(distances, batch_labels)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)

            history['loss'].append(avg_train_loss)
            history['distance_accuracy'].append(avg_train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_distance_accuracy'].append(avg_val_acc)

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        return history

    # -------------------------------
    # Reference Embeddings
    # -------------------------------
    def extract_reference_embeddings(self, food_data, samples_per_class=10):
        all_images, all_labels, all_calories = [], [], []
        categories_map = {}

        for img, label in zip(food_data.get("train_images", []), food_data.get("train_labels", [])):
            cal = food_data.get("train_cal", {}).get(label, 0)
            if label not in categories_map:
                categories_map[label] = {'train': [], 'val': []}
            categories_map[label]['train'].append((img, cal))
        for img, label in zip(food_data.get("val_images", []), food_data.get("val_labels", [])):
            cal = food_data.get("val_cal", {}).get(label, 0)
            if label not in categories_map:
                categories_map[label] = {'train': [], 'val': []}
            categories_map[label]['val'].append((img, cal))

        for category, samples in categories_map.items():
            category_images = []
            category_calories = []
            train_samples = samples['train']
            selected_train = random.sample(train_samples, min(samples_per_class, len(train_samples)))
            category_images.extend([s[0] for s in selected_train])
            category_calories.extend([s[1] for s in selected_train])
            val_samples = samples['val']
            needed = samples_per_class - len(category_images)
            if needed > 0 and val_samples:
                selected_val = random.sample(val_samples, min(needed, len(val_samples)))
                category_images.extend([s[0] for s in selected_val])
                category_calories.extend([s[1] for s in selected_val])
            for img, cal in zip(category_images, category_calories):
                all_images.append(img)
                all_labels.append(category)
                all_calories.append(cal)

        if not all_images:
            print("No reference images found.")
            return []

        all_images = np.array(all_images).astype('float32')
        preprocessed = self._preprocess_images_for_vgg(all_images)

        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            embeddings = self.forward_one(preprocessed.to(device)).cpu().numpy()
        for emb, label, cal in zip(embeddings, all_labels, all_calories):
            if label not in self.reference_embeddings:
                self.reference_embeddings[label] = []
            self.reference_embeddings[label].append((emb, cal))
        print(f"Extracted {len(embeddings)} reference embeddings from {len(categories_map)} categories.")
        return list(categories_map.keys())

    # -------------------------------
    # Prediction for single image
    # -------------------------------
    def predict_food_type(self, image_path, threshold=0.5):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        img_resized = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img_arr = np.expand_dims(img_resized.astype('float32'), axis=0)
        img_pre = self._preprocess_images_for_vgg(img_arr)
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            query_embedding = self.forward_one(img_pre.to(device)).cpu().numpy()[0]
        distances, categories, avg_calories = [], [], []
        for category, refs in self.reference_embeddings.items():
            dists = [np.linalg.norm(query_embedding - ref[0]) for ref in refs]
            cals = [ref[1] for ref in refs if ref[1] > 0]
            distances.append(np.mean(dists))
            categories.append(category)
            avg_calories.append(np.mean(cals) if cals else 0)
        if not distances:
            return {'category':'Unknown','confidence':0.0,'calories_per_gram':0.0,'all_predictions':[]}
        best_idx = int(np.argmin(distances))
        confidence = 1.0 / (1.0 + distances[best_idx])
        best_category = categories[best_idx] if confidence >= threshold else 'Unknown'
        all_predictions = [{'category':cat,'distance':float(dist),'confidence':1.0/(1.0+float(dist)),'calories_per_gram':cal}
                           for cat, dist, cal in zip(categories, distances, avg_calories)]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return {'category':best_category,'confidence':float(confidence),
                'calories_per_gram':float(avg_calories[best_idx]),'all_predictions':all_predictions}

    # -------------------------------
    # Save/Load
    # -------------------------------
    def save_model(self, path='partB_siamese_model'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'model_state_dict': self.state_dict(),'input_shape': self.input_shape}, f'{path}_model.pth')
        with open(f'{path}_embeddings.pkl','wb') as f:
            pickle.dump(self.reference_embeddings, f)
        print(f"Model saved to {path}_model.pth. Embeddings saved to {path}_embeddings.pkl.")

    def load_model(self, path='partB_siamese_model'):
        checkpoint = torch.load(f'{path}_model.pth', map_location='cpu')
        self.__init__(checkpoint.get('input_shape',(224,224,3)))
        self.load_state_dict(checkpoint['model_state_dict'])
        with open(f'{path}_embeddings.pkl','rb') as f:
            self.reference_embeddings = pickle.load(f)
        print(f"Model loaded from {path}_model.pth. Embeddings loaded from {path}_embeddings.pkl.")

# -------------------------------
# Runner
# -------------------------------
def run_partB(food_data, epochs=10, batch_size=16):
    print("="*60)
    print("PART B: Food Recognition using Siamese Network (VGG16 encoder, contrastive loss)")
    print("="*60)
    model = SiameseNetwork()
    history = model.train_model(food_data, epochs=epochs, batch_size=batch_size)
    categories = model.extract_reference_embeddings(food_data, samples_per_class=10)

    # Plot
    try:
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(history['distance_accuracy'], label='Train Acc')
        plt.plot(history['val_distance_accuracy'], label='Val Acc')
        plt.title('Distance Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend(); plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("Could not plot:", e)

    model.save_model()
    print(f"\nTrained on {len(categories)} categories. Model ready!")
    return model, lambda img_path: model.predict_food_type(img_path)
