from ultralytics import YOLO

model = YOLO("./models/best_100.pt")

results = model.predict(source="./test-images/crackk.webp", conf=0.4, device="mps", imgsz=640, save=True)
