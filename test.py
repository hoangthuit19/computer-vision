import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load model YOLO và OCR
yolo_model = YOLO("models/text_detector.pt")
ocr_model = PaddleOCR(use_angle_cls=True, lang='vi')  # OCR tiếng Việt

def detect_and_read(image_path):
    results = yolo_model(image_path)
    image = cv2.imread(image_path)

    output = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()   # Tọa độ [x1, y1, x2, y2]
        scores = r.boxes.conf.cpu().numpy()  # Confidence YOLO

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]

            # OCR trên crop
            ocr_result = ocr_model.ocr(crop, cls=True)

            if ocr_result and ocr_result[0]:
                text, conf = ocr_result[0][0][1][0], ocr_result[0][0][1][1]
                output.append({
                    "bbox": [x1, y1, x2, y2],
                    "yolo_conf": float(score),
                    "text": text,
                    "ocr_conf": float(conf)
                })

    return output

# --- Test ---
results = detect_and_read("test.jpg")
for r in results:
    print(r)
