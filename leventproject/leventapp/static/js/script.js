const videoElement = document.getElementById("camera");
const canvasElement = document.getElementById("outputCanvas");
const ctx = canvasElement.getContext("2d");

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  videoElement.srcObject = stream;

  const sendFrame = async () => {
    const canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg").split(",")[1]; // Base64 format

    // POST isteği
    try {
      const response = await fetch("/process-frame/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
      });

      const result = await response.json();
      if (result.status === "success") {
        const img = new Image();
        img.src = `data:image/jpeg;base64,${result.image}`;
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        };
      }
    } catch (error) {
      console.error("Hata:", error);
    }
    requestAnimationFrame(sendFrame);
  };

  sendFrame();
}

document.addEventListener("DOMContentLoaded", startCamera);
