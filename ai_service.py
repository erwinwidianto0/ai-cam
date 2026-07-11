import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import sys
import os

app = FastAPI(title="YOLOv8 Inference Service", description="Local AI service for CCTV object detection")

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from huggingface_hub import hf_hub_download

# Memuat model secara Hybrid (Base Model COCO + Custom Model + Fire/Smoke Model + Florence-2 VLM)
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
    
    print("Loading Florence-2 local VLM model from Microsoft...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    florence_model_id = "microsoft/Florence-2-base"
    florence_model = AutoModelForCausalLM.from_pretrained(florence_model_id, trust_remote_code=True).eval().to(device)
    florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)
    print(f"Florence-2 local VLM loaded successfully on device: {device}")
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
            results_person = model_person.track(image, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.15)
        except Exception as e_track:
            print(f"Tracking failed, falling back to standard detection: {e_track}")
            results_person = model_person(image, verbose=False, conf=0.15)
        
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

# Helper untuk memproses tugas VLM menggunakan Microsoft Florence-2 secara lokal
def run_florence_task(image, task_prompt, text_input=None):
    if text_input:
        prompt = task_prompt + text_input
    else:
        prompt = task_prompt
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=1  # Menggunakan greedy search (1 beam) agar 3x lebih cepat di CPU
        )
        
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

@app.post("/vlm/local")
async def vlm_local(file: bytes = File(...)):
    try:
        # Baca gambar dari byte raw JPEG
        image = Image.open(io.BytesIO(file))
        
        # 1. Dapatkan Deskripsi Detail (Detailed Caption) - 1 call
        caption_result = run_florence_task(image, "<MORE_DETAILED_CAPTION>")
        description = caption_result.get("<MORE_DETAILED_CAPTION>", "No description available.")
        
        # 2. Dapatkan deteksi objek kognitif gabungan (Phrase Grounding) - 1 call (total hanya 2 call model)
        grounding_result = run_florence_task(
            image, 
            "<CAPTION_TO_PHRASE_GROUNDING>", 
            "fire, smoke, person smoking, cigarette, person sleeping, person lying down, mess on floor, items on floor, person, stove, kitchen cabinet"
        )
        
        parsed_grounding = grounding_result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        labels = parsed_grounding.get("labels", [])
        
        # Analisis label secara logis (ubah ke lowercase)
        labels_lower = [l.lower() for l in labels]
        desc_lower = description.lower()
        
        # Deteksi Kebakaran & Asap Hibrida (Grounding + Caption Keywords)
        fire_detected = any("fire" in l or "api" in l or "flame" in l for l in labels_lower) or any(w in desc_lower for w in ["fire", "kebakaran", "flame", "burning"])
        smoke_detected = any("smoke" in l or "asap" in l for l in labels_lower) or any(w in desc_lower for w in ["smoke", "asap", "kepulan asap"])
        
        # Deteksi Rokok Hibrida (Grounding + Caption Keywords)
        smoking_detected = any("smoking" in l or "cigarette" in l or "rokok" in l for l in labels_lower) or any(w in desc_lower for w in ["smoking", "cigarette", "rokok", "merokok"])
        
        # Deteksi Petugas Tidur Hibrida (Grounding + Caption Keywords)
        sleeping_detected = any("sleeping" in l or "lying down" in l or "tidur" in l for l in labels_lower) or any(w in desc_lower for w in ["sleeping", "lying down", "tidur", "tertidur", "is asleep"])
        
        # Deteksi Pelanggaran SOP Hibrida
        sop_violation_detected = any("mess" in l or "floor" in l or "trash" in l for l in labels_lower) or any(w in desc_lower for w in ["mess on floor", "trash", "dirty kitchen", "littering", "items on floor"])
        
        # Evaluasi Memasak secara kognitif: manusia berada di dekat kompor/lemari dapur
        cooking_detected = "person" in labels_lower and ("stove" in labels_lower or "kitchen cabinet" in labels_lower)

        return JSONResponse(content={
            "cooking": cooking_detected,
            "fire": fire_detected or smoke_detected,
            "smoking": smoking_detected,
            "sleeping": sleeping_detected,
            "sop_violation": sop_violation_detected,
            "description": description
        })
        
    except Exception as e:
        print(f"Error processing local VLM task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Jalankan server lokal di port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
