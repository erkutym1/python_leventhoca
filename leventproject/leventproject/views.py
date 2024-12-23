from django.shortcuts import render

from django.conf import settings


def camera_view(request):
    return render(request, 'camera.html')


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import base64
import mediapipe as mp

import torch
import torch.nn as nn
import time

# Mediapipe yapılandırması
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

LABELS = [chr(i) for i in range(65, 91)]  # A-Z
input_size = 21 * 3  # 21 landmark (x, y, z)

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


def is_hand_stationary(current_landmarks, previous_landmarks, threshold=0.02):
    if previous_landmarks is None:
        return False
    # Landmarklar arasındaki ortalama farkı hesapla
    diffs = np.linalg.norm(np.array(current_landmarks) - np.array(previous_landmarks), axis=1)
    return np.mean(diffs) < threshold

def predict_landmarks(landmarks, model, device):
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(landmarks_tensor)
        predicted_class = output.argmax(1).item()
    return LABELS[predicted_class]


@csrf_exempt  # CSRF korumasını geçici olarak devre dışı bırakıyoruz (güvenliğe dikkat edin!)
def process_image(request):
    if request.method == 'POST':
        # Görüntüyü al
        image_data = request.POST.get('image')  # Base64 formatında
        image_data = image_data.split(',')[1]  # Data URL kısmını ayır
        image = base64.b64decode(image_data)

        # Görüntüyü aç
        np_array = np.frombuffer(image, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        frame1 = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Görüntüyü yatay olarak çevir (flip)
        frame_flipped = cv2.flip(rgb_frame, 1)

        frame = frame_flipped.copy()

        # Modeli yükle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(LABELS)
        model = ASLLandmarkModel(input_size, num_classes).to(device)
        MODEL_PATH = settings.MODEL_PATH
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model başarıyla yüklendi.")

        previous_landmarks = None
        predicted_label = ""

        result = hands.process(frame)

        # El işaretlerini tanı
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Landmark koordinatlarını çıkar
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                # El işaretini tahmin et
                predicted_label = predict_landmarks([coord for lm in landmarks for coord in lm], model, device)

                # Landmarkları çiz
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                previous_landmarks = landmarks
        else:
            cv2.putText(frame, "El algilanamadi!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            previous_landmarks = None  # Landmarkları sıfırla

        _, buffer_flipped = cv2.imencode('.jpg', frame_flipped)
        encoded_flipped = base64.b64encode(buffer_flipped).decode('utf-8')

        _, buffer_rgb = cv2.imencode('.jpg', rgb_frame)
        encoded_rgb = base64.b64encode(buffer_rgb).decode('utf-8')

        # Marker görüntüsünü base64'e çevir
        _, buffer_marker = cv2.imencode('.jpg', frame)
        encoded_marker = base64.b64encode(buffer_marker).decode('utf-8')

        return JsonResponse({
            'resultImage2': f'data:image/jpeg;base64,{encoded_rgb}',  # RGB görüntü
            'resultImage3': f'data:image/jpeg;base64,{encoded_flipped}',  # Yatay flip sonrası
            'resultImage4': f'data:image/jpeg;base64,{encoded_marker}',  # Marker görüntü
            'tahmin': predicted_label  # El işareti tahmini
        })
    return JsonResponse({'error': 'Invalid request method'}, status=400)






@csrf_exempt
def oyun_sayfasi(request):
    return render(request, 'oyun_sayfasi.html')

@csrf_exempt
def camera_page(request):
    return render(request, 'camera.html')

@csrf_exempt
def foto_page(request):
    return render(request, 'foto_page.html')




from django.http import JsonResponse
from django.shortcuts import render
from .models import TextModel  # textmodel yerine TextModel

def save_data(request):
    if request.method == 'POST':
        texts = request.POST.get('inputText')
        if texts:
            TextModel.objects.create(texts=texts)
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'}, status=400)

def fetch_data(request):
    data = TextModel.objects.all().values_list('texts', flat=True)
    content = "\n".join(data)  # Convert the list of texts into a single string
    return JsonResponse(content, safe=False)


