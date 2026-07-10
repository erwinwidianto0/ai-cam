import sys
from ultralytics import YOLO

print("Memuat base model yolov8n.pt...")
model = YOLO("yolov8n.pt")

print("Mulai proses pelatihan...")
model.train(
    data="./dataset/data.yaml",
    epochs=50,
    imgsz=640,
    device="cpu",
    verbose=True
)
print("TRAINING_COMPLETED_SUCCESSFULLY")
