import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

class FoodFruitDataset(Dataset):
    def __init__(self, food_images, fruit_images):
        self.images = np.concatenate([food_images, fruit_images])
        self.labels = np.concatenate([
            np.zeros(len(food_images)),
            np.ones(len(fruit_images))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # HWC → CHW
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label



class PartAModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze()




def train_partA(model, train_loader, val_loader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (outputs > 0.5).float()

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = val_correct / val_total

        history['loss'].append(running_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss:.4f} Acc: {train_acc:.4f} "
              f"Val_Loss: {val_loss:.4f} Val_Acc: {val_acc:.4f}")

    return history, np.array(y_true), np.array(y_pred)





def evaluate_partA(y_true, y_pred):
    print("\n" + "="*50)
    print("PART A - Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=["Food", "Fruit"]))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Food", "Fruit"],
                yticklabels=["Food", "Fruit"])
    # plt.show()




def classify_food_fruit(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(img).item()

    return {
        "classification": "Fruit" if prob > 0.5 else "Food",
        "confidence": prob if prob > 0.5 else 1 - prob,
        "probability_fruit": prob,
        "probability_food": 1 - prob
    }


# NEW FUNCTION: Main execution function for Part A
def run_partA(food_data, fruit_data, use_generator=True, batch_size=16, epochs=20):
    """
    Main function to run Part A training and evaluation
    """
    print("\nPreparing Part A Dataset...")
    
    # Prepare training data
    train_food_images = food_data["train_images"]
    train_fruit_images = fruit_data["train_images"]
    
    # Prepare validation data
    val_food_images = food_data["val_images"]
    val_fruit_images = fruit_data["val_images"]
    
    # Create datasets
    train_dataset = FoodFruitDataset(train_food_images, train_fruit_images)
    val_dataset = FoodFruitDataset(val_food_images, val_fruit_images)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True   
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True   
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = PartAModel()
    print("\nModel architecture:")
    print(model)
    
    # Train the model
    print(f"\nTraining for {epochs} epochs...")
    history, y_true, y_pred = train_partA(
        model, train_loader, val_loader, 
        epochs=epochs, lr=1e-3
    )
    
    # Evaluate the model
    evaluate_partA(y_true, y_pred)
    
    # Save the model
    torch.save(model.state_dict(), "partA_food_fruit_classifier.pth")
    print("Model saved as 'partA_food_fruit_classifier.pth'")
    
    # Create test function
    def test_function(image_path):
        return classify_food_fruit(model, image_path)
    
    return model, history, test_function


# Optional: Add this for direct testing
if __name__ == "__main__":
    # This would require loading data first
    print("This module should be imported and used from main.py")