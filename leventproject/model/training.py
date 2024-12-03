import os
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# Mediapipe yapılandırması
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Veri kümesi dizini ve etiketler
DATASET_DIR = "../../DATASET/ASL_Alphabet_Dataset/asl_alphabet_train"
LABELS = [chr(i) for i in range(65, 91)]  # A-Z

# Landmark verilerini ve etiketleri saklayacak listeler
landmark_data = []
landmark_labels = []

print("Landmark verileri işleniyor...")
for label in LABELS:
    label_dir = os.path.join(DATASET_DIR, label)
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe ile el landmarklarını çıkar
        result = hands.process(image_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Her landmark'ın (x, y, z) koordinatlarını çıkar
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # x, y, z koordinatları
                landmark_data.append(landmarks)
                landmark_labels.append(label)

print("Landmark verileri işleme tamamlandı.")

# Landmark verilerini ve etiketleri bir DataFrame'e dönüştür
df = pd.DataFrame(landmark_data)
df['label'] = landmark_labels

# Verileri kaydet
df.to_csv("landmark_dataset.csv", index=False)
print("Veriler landmark_dataset.csv dosyasına kaydedildi.")


class LandmarkDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class ASLLandmarkModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ASLLandmarkModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# Landmark verilerini ve etiketlerini yükle
df = pd.read_csv("landmark_dataset.csv")
X = df.iloc[:, :-1].values
y = LabelEncoder().fit_transform(df['label'])

# Eğitim ve doğrulama setlerine ayırma
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri kümesi ve veri yükleyiciler
train_dataset = LandmarkDataset(X_train, y_train)
val_dataset = LandmarkDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, kriter ve optimizasyon
input_size = X.shape[1]
num_classes = len(LABELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLLandmarkModel(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model eğitimi
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=15):
    for epoch in range(epochs):
        # Eğitim modu
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == targets).sum().item()

        # Doğrulama modu
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_correct += (outputs.argmax(1) == targets).sum().item()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_correct / len(train_dataset):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_correct / len(val_dataset):.4f}")

# Modeli eğit ve kaydet
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=15)
torch.save(model.state_dict(), "landmark_model.pth")
print("Model landmark_model.pth olarak kaydedildi.")


