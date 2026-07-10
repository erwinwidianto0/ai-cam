import sys
from ultralytics import YOLO

print("Memuat base model yolo11n.pt...")
model = YOLO("yolo11n.pt")

print("Mulai proses pelatihan...")
model.train(
    data="./dataset/data.yaml",
    epochs=30,
    imgsz=640,
    device="cpu",
    verbose=True
)
print("TRAINING_COMPLETED_SUCCESSFULLY")
