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

@app.post("/reload")
def reload_model():
    global model_custom
    try:
        if os.path.exists("custom_model.pt"):
            print("Reloading custom model...")
            model_custom = YOLO("custom_model.pt")
            print("Custom model reloaded successfully.")
            return {"status": "ok", "message": "Custom model reloaded successfully."}
        else:
            return {"status": "error", "message": "custom_model.pt not found."}
    except Exception as e:
        print(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect(file: bytes = File(...)):
    try:
        # Baca gambar dari byte raw JPEG
        image = Image.open(io.BytesIO(file))
        
        # 1. Jalankan inferensi YOLOv8 dengan ByteTrack (Object Tracking) untuk melacak objek secara kontinu
        try:
            results_person = model_person.track(image, persist=True, tracker="bytetrack.yaml", verbose=False)
        except Exception as e_track:
            print(f"Tracking failed, falling back to standard detection: {e_track}")
            results_person = model_person(image, verbose=False)
        
        # 2. Jalankan inferensi YOLOv8 untuk api/asap
        results_fire = model_fire(image, verbose=False)
        
        detections = []
        
        # 1. Proses deteksi objek standar dari base model (COCO 80 kelas)
        for result in results_person:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model_person.names[cls_id]
                
                # Kamus terjemahan COCO ke Bahasa Indonesia untuk UI dasbor Anda
                COCO_TRANSLATION = {
                    "person": "person",  # Go melacak area memasak dengan kata kunci "person"
                    "car": "mobil",
                    "motorcycle": "motor",
                    "bicycle": "sepeda",
                    "backpack": "tas",
                    "umbrella": "payung",
                    "handbag": "tas tangan",
                    "tie": "dasi",
                    "suitcase": "koper",
                    "bottle": "botol",
                    "wine glass": "gelas",
                    "cup": "cangkir/gelas",
                    "fork": "garpu",
                    "knife": "pisau",
                    "spoon": "sendok",
                    "bowl": "mangkuk",
                    "chair": "kursi",
                    "couch": "sofa",
                    "potted plant": "tanaman",
                    "bed": "kasur",
                    "dining table": "meja",
                    "tv": "tv",
                    "laptop": "laptop",
                    "mouse": "mouse",
                    "keyboard": "keyboard",
                    "cell phone": "hp",
                    "microwave": "microwave",
                    "oven": "oven",
                    "sink": "wastafel",
                    "refrigerator": "kulkas",
                    "book": "buku",
                    "clock": "jam",
                    "scissors": "gunting"
                }
                
                mapped_label = COCO_TRANSLATION.get(label, label)
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                det_item = {
                    "class": cls_id,
                    "label": mapped_label,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                }
                
                # Tambahkan ID tracking unik jika objek dilacak
                if box.id is not None:
                    det_item["track_id"] = int(box.id[0])
                    
                detections.append(det_item)
                    
        # 2. Jalankan inferensi model kustom jika terpasang (untuk kelas kustom unik Anda)
        if model_custom is not None:
            try:
                results_custom = model_custom.track(image, persist=True, tracker="bytetrack.yaml", verbose=False)
            except Exception as e_custom_track:
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
                        
                        det_item = {
                            "class": cls_id + 200, # Offset agar unik
                            "label": mapped_label,
                            "confidence": confidence,
                            "box": [x1, y1, x2, y2]
                        }
                        
                        # Tambahkan ID tracking unik jika objek dilacak
                        if box.id is not None:
                            det_item["track_id"] = int(box.id[0])
                            
                        detections.append(det_item)
                        
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
