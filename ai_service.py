import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import sys
import os

app = FastAPI(title="YOLOv8 Inference Service", description="Local AI service for CCTV object detection")

from huggingface_hub import hf_hub_download

# Memuat model (yolo26n.pt (YOLO26) untuk manusia, rabahdev/fire-smoke-yolov8n untuk api/asap)
try:
    print("Loading YOLO26 person model...")
    model_path = "custom_model.pt" if os.path.exists("custom_model.pt") else "yolo26n.pt"
    print(f"Using person model path: {model_path}")
    model_person = YOLO(model_path)
    print("YOLO26 person model loaded successfully.")
    
    print("Downloading/Loading YOLOv8 fire/smoke model from Hugging Face...")
    fire_ckpt = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
    model_fire = YOLO(fire_ckpt)
    print("YOLOv8 fire/smoke model loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

@app.get("/health")
def health():
    return {"status": "ok", "person_model": "yolov8n.pt", "fire_model": "fire-smoke-yolov8n"}

@app.post("/detect")
async def detect(file: bytes = File(...)):
    try:
        # Baca gambar dari byte raw JPEG
        image = Image.open(io.BytesIO(file))
        
        # Jalankan inferensi YOLOv8 untuk manusia
        results_person = model_person(image, verbose=False)
        
        # Jalankan inferensi YOLOv8 untuk api/asap
        results_fire = model_fire(image, verbose=False)
        
        detections = []
        
        # 1. Proses deteksi manusia
        for result in results_person:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model_person.names[cls_id]
                
                # Hanya ambil objek manusia ("person")
                if label == "person":
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    detections.append({
                        "class": cls_id,
                        "label": label,
                        "confidence": confidence,
                        "box": [x1, y1, x2, y2]
                    })
                    
        # 2. Proses deteksi api/asap
        for result in results_fire:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model_fire.names[cls_id]
                
                # Hanya ambil objek api ("fire") atau asap ("smoke")
                if label in ["fire", "smoke"]:
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    detections.append({
                        "class": cls_id + 100, # Offset ID agar unik
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
