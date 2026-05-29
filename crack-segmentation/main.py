from ultralytics import YOLO



model = YOLO("yolo26n-seg.pt")


results = model.train(data="./datasets/crack-seg.yaml", epochs=100, imgsz=640, device="mps")


model.save("yolov26-seg-crack.pt")