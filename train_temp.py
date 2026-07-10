import sys
from ultralytics import YOLO

print("Memuat base model yolo26n.pt...")
model = YOLO("yolo26n.pt")

print("Mulai proses pelatihan...")
model.train(
    data="./dataset/data.yaml",
    epochs=30,
    imgsz=640,
    device="cpu",
    verbose=True
)
print("TRAINING_COMPLETED_SUCCESSFULLY")
