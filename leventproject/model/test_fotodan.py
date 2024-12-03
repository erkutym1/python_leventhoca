import cv2
import mediapipe as mp
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch.nn as nn

# Etiketleri yükleme (A-Z)
LABELS = [chr(i) for i in range(65, 91)]  # A-Z

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

# Model ve aygıt tanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 21 * 3  # 21 landmark * (x, y, z)
num_classes = len(LABELS)
model = ASLLandmarkModel(input_size, num_classes).to(device)

# Eğitilmiş modeli yükleme
MODEL_PATH = "landmark_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model başarıyla yüklendi.")

# Mediapipe yapılandırması
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def predict_image(image_path, model, device):
    # Görseli yükle ve RGB'ye dönüştür
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mediapipe ile el landmarklarını çıkar
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Landmark koordinatlarını çıkar
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # x, y, z

            # Landmarkları tensora dönüştür
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)

            # Model ile tahmin yap
            with torch.no_grad():
                output = model(landmarks_tensor)
                predicted_class = output.argmax(1).item()

            # Tahmini döndür
            return LABELS[predicted_class]
    else:
        return "El algılanamadı!"


# Test edilecek görsel
TEST_IMAGE_PATH = "testfoto.jpg"

# Tahmin yap
predicted_label = predict_image(TEST_IMAGE_PATH, model, device)
print(f"Tahmin edilen harf: {predicted_label}")

# Görseli göster
image = cv2.imread(TEST_IMAGE_PATH)
cv2.putText(image, f"Tahmin: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Tahmin Sonucu", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
