const videoElement1 = document.getElementById('camera1');
const videoElement2 = document.getElementById('camera2');
const videoElement3 = document.getElementById('camera3');
const videoElement4 = document.getElementById('camera4');

async function startCamera() {
  try {
    // Kamera erişimi için izin al
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement1.srcObject = stream;
    videoElement1.style.transform = 'scaleX(-1)';
    videoElement1.style.width = '30%';

    videoElement2.srcObject = stream;
    videoElement2.style.width = '22%';

    videoElement3.srcObject = stream;
    videoElement3.style.transform = 'scaleX(-1)';
    videoElement3.style.width = '22%';

    videoElement4.srcObject = stream;
    videoElement4.style.transform = 'scaleX(-1)';
    videoElement4.style.width = '22%';


  } catch (error) {
    console.error("Kamera erişimi sağlanamadı:", error);
    alert("Kamera erişimi sağlanamadı. Lütfen tarayıcı ayarlarını kontrol edin.");
  }
}

// Sayfa yüklendiğinde kamerayı başlat
window.addEventListener('load', startCamera);
