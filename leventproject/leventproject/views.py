from django.shortcuts import render

def camera_view(request):
    return render(request, 'camera.html')


from django.shortcuts import render
from django.http import JsonResponse
import cv2
import numpy as np
import torch
import torch.nn as nn
import base64
from io import BytesIO
from PIL import Image
import json
from django.conf import settings

# Model giriş bilgileri ve etiketler
LABELS = [chr(i) for i in range(65, 91)]  # A-Z
input_size = 21 * 3  # 21 landmark (x, y, z)

# Model sınıfı
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

# Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLLandmarkModel(input_size, len(LABELS)).to(device)
model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
model.eval()

# Tahmin fonksiyonu
def predict_landmarks(landmarks, model, device):
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(landmarks_tensor)
        predicted_class = output.argmax(1).item()
    return LABELS[predicted_class]

# Görüntü işleme ve tahmin için view
def predict_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data.get('image')

        # Base64 formatından OpenCV görüntüsüne çevirme
        image_bytes = BytesIO(base64.b64decode(image_data.split(',')[1]))
        image = Image.open(image_bytes)
        frame = np.array(image)

        # Görüntüyü RGB'ye çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe ile el algılama
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                prediction = predict_landmarks([coord for lm in landmarks for coord in lm], model, device)
                return JsonResponse({'prediction': prediction})

        return JsonResponse({'prediction': 'El algılanamadı'})
    else:
        return JsonResponse({'error': 'Geçersiz istek'}, status=400)

# Ana sayfa
def index(request):
    return render(request, 'index.html')
