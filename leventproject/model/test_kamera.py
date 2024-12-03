import cv2
import mediapipe as mp
import torch
import numpy as np
import torch.nn as nn
import time

# Model giriş bilgileri ve etiketler
LABELS = [chr(i) for i in range(65, 91)]  # A-Z
input_size = 21 * 3  # 21 landmark (x, y, z)

# Mediapipe yapılandırması
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Landmark tabanlı model sınıfı
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

# Model ve aygıt ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(LABELS)
model = ASLLandmarkModel(input_size, num_classes).to(device)

# Modeli yükle
MODEL_PATH = "landmark_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model başarıyla yüklendi.")

# Tahmin fonksiyonu
def predict_landmarks(landmarks, model, device):
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(landmarks_tensor)
        predicted_class = output.argmax(1).item()
    return LABELS[predicted_class]

# Hareket kontrolü
def is_hand_stationary(current_landmarks, previous_landmarks, threshold=0.02):
    if previous_landmarks is None:
        return False
    # Landmarklar arasındaki ortalama farkı hesapla
    diffs = np.linalg.norm(np.array(current_landmarks) - np.array(previous_landmarks), axis=1)
    return np.mean(diffs) < threshold

# Kamera ile tahmin
cap = cv2.VideoCapture(0)
print("Kamera açıldı. Çıkış için 'q' tuşuna basın.")

previous_landmarks = None
stationary_start_time = None
stationary_duration_threshold = 0.7  # Hareketsizlik süresi (saniye)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera verisi okunamadı!")
        break

    # Görüntüyü yatay çevir (flip)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe ile el algılama
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Landmark koordinatlarını çıkar
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            # Landmarkların hareketsiz olup olmadığını kontrol et
            if is_hand_stationary(landmarks, previous_landmarks):
                if stationary_start_time is None:
                    stationary_start_time = time.time()  # Hareketsizlik başladı
                elif time.time() - stationary_start_time > stationary_duration_threshold:
                    # Hareketsizlik süresi geçti, tahmin yap
                    predicted_label = predict_landmarks([coord for lm in landmarks for coord in lm], model, device)
                    cv2.putText(frame, f"Tahmin: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                stationary_start_time = None  # Hareketsizlik sona erdi

            # Landmarkları çiz
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            previous_landmarks = landmarks
    else:
        cv2.putText(frame, "El algilanamadi!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        previous_landmarks = None  # Landmarkları sıfırla
        stationary_start_time = None

    # Görüntüyü göster
    cv2.imshow("ASL Tahmin", frame)

    # 'q' tuşuyla çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
