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

# Memuat model secara Hybrid (Base Model COCO + Custom Model + Fire/Smoke Model)
try:
    print("Loading YOLO26 base model...")
    model_person = YOLO("yolo26n.pt")
    print("YOLO26 base model loaded successfully.")
    
    model_custom = None
    if os.path.exists("custom_model.pt"):
        print("Loading custom model...")
        model_custom = YOLO("custom_model.pt")
        print("Custom model loaded successfully.")
    else:
        print("Custom model not found. Using base model only.")
        
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
        
        # 1. Jalankan inferensi YOLOv8 untuk manusia/objek standar (menggunakan base model COCO yang super akurat)
        results_person = model_person(image, verbose=False)
        
        # 2. Jalankan inferensi YOLOv8 untuk api/asap
        results_fire = model_fire(image, verbose=False)
        
        detections = []
        
        # 1. Proses deteksi objek standar (person/car/motorcycle) dari base model
        for result in results_person:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model_person.names[cls_id]
                
                # Gunakan model COCO untuk mendeteksi manusia, mobil, dan motor demi akurasi tinggi
                if label in ["person", "car", "motorcycle"]:
                    mapped_label = label
                    if label == "car":
                        mapped_label = "mobil"
                    elif label == "motorcycle":
                        mapped_label = "motor"
                    elif label == "person":
                        mapped_label = "person" # Go mengharapkan label "person"
                        
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    detections.append({
                        "class": cls_id,
                        "label": mapped_label,
                        "confidence": confidence,
                        "box": [x1, y1, x2, y2]
                    })
                    
        # 2. Jalankan inferensi model kustom jika terpasang (untuk kelas kustom unik Anda)
        if model_custom is not None:
            results_custom = model_custom(image, verbose=False)
            for result in results_custom:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = model_custom.names[cls_id]
                    
                    # Hanya ambil kelas kustom baru, lewati manusia/mobil/motor agar tidak duplikasi
                    if label not in ["manusia", "mobil", "motor"]:
                        mapped_label = label
                        if label == "api":
                            mapped_label = "fire"
                        elif label == "asap":
                            mapped_label = "smoke"
                            
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        detections.append({
                            "class": cls_id + 200, # Offset agar unik
                            "label": mapped_label,
                            "confidence": confidence,
                            "box": [x1, y1, x2, y2]
                        })
                        
        # 3. Proses deteksi api/asap dari model khusus fire/smoke
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
