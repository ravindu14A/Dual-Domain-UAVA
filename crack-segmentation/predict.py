from ultralytics import YOLO
import cv2
import time
import numpy as np


def draw_segmentation(frame, results, model):
    overlay = frame.copy()

    # Draw segmentation masks
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()  # (N, H, W)
        for i, mask in enumerate(masks):
            # Resize mask to frame size
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_bool = mask_resized > 0.5

            # Random-ish color per class
            cls_id = int(results.boxes.cls[i])
            color = [
                int((cls_id * 67 + 80) % 255),
                int((cls_id * 113 + 160) % 255),
                int((cls_id * 191 + 40) % 255),
            ]
            overlay[mask_bool] = color

    # Blend overlay with original
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Draw boxes + labels on top
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 2), (x1 + tw, y1), (0, 255, 80), -1)
        cv2.putText(frame, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def main():
    model = YOLO("./models/best_100.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    # Try to set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, device="mps", conf=0.65, imgsz=640)[0]

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        frame = draw_segmentation(frame, results, model)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)

        cv2.imshow("Crack Segmentation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()