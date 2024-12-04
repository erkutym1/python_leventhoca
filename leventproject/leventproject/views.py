from django.shortcuts import render

def camera_view(request):
    return render(request, 'camera.html')


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import base64
import mediapipe as mp

# Mediapipe yapılandırması
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


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

        # Görüntüyü yatay olarak çevir (flip)
        frame = cv2.flip(frame, 1)

        # Görüntüyü RGB formatına çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Landmark'ları çizme
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Sonuç görüntüsünü base64 formatına çevir
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_result = base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({'image': f'data:image/jpeg;base64,{encoded_result}'})
    return JsonResponse({'error': 'Invalid request method'}, status=400)
