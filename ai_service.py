import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import sys

app = FastAPI(title="YOLOv8 Inference Service", description="Local AI service for CCTV object detection")

# Memuat model YOLOv8 (yolov8n.pt adalah model paling ringan dan cepat)
# Model akan diunduh secara otomatis pada saat pertama kali dijalankan jika belum ada.
try:
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8n.pt"}

@app.post("/detect")
async def detect(file: bytes = File(...)):
    try:
        # Baca gambar dari byte raw JPEG
        image = Image.open(io.BytesIO(file))
        
        # Jalankan inferensi YOLOv8
        # verbose=False untuk mengurangi output log yang tidak perlu di konsol
        results = model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])
                # Koordinat bounding box x1, y1, x2, y2
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                detections.append({
                    "class": cls_id,
                    "label": label,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                })
                
        return JSONResponse(content={"detections": detections})
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Jalankan server lokal di port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
