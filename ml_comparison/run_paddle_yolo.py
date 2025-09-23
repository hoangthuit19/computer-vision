import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from main_ocr import SimpleOCRProcessor
from accuracy import AccuracyEvaluator
import time

# Load YOLO model đã train
yolo_model = YOLO("../models/text_detector.pt")
# Load PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='vi')

def run_pipeline(image_path, processor=None):
    # image = cv2.imread(image_path)
    
    if processor is not None:
        print("[INFO] Running preprocessing with processor...")
        processed_images = processor.visualize_processing_steps(image_path)
        if "preprocessed" in processed_images:
            print("[INFO] Using preprocessed image...")
            image = processed_images["preprocessed"]
        else:
            image = cv2.imread(image_path)
    else:
        image = cv2.imread(image_path)
    
    results = yolo_model(image)
    
    results[0].show()

    outputs = []
    start_time = time.time()
    for r in results:
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0].item())
            class_name = yolo_model.names[cls_id]

            # Crop vùng bbox từ YOLO
            crop = image[y1:y2, x1:x2]

            # OCR với PaddleOCR
            ocr_result = ocr.predict(crop)
            text = ""
            if ocr_result and len(ocr_result) > 0:
                rec_texts = ocr_result[0].get("rec_texts", [])
                rec_scores = ocr_result[0].get("rec_scores", [])
                if rec_texts:
                    # lấy text có confidence cao nhất
                    best_idx = int(max(range(len(rec_scores)), key=lambda i: rec_scores[i]))
                    text = rec_texts[best_idx]

            outputs.append({
                "class": class_name,
                "ocr_text": text,
                "bbox": [x1, y1, x2, y2]
            })
    end_time = time.time()
    print(f"⏱ Tổng thời gian xử lý: {end_time - start_time:.2f} giây")
    
    evaluator = AccuracyEvaluator("../data.csv")
        
    acc = evaluator.calculate_accuracy(outputs, "dat_016")
    print(f"Accuracy: {acc:.2f}%")

    return outputs

# --- Test ---
if __name__ == "__main__":
    processor = SimpleOCRProcessor()
    
    img = "../test_img/dat_016.jpg"
    results = run_pipeline(img, processor=processor)
    for r in results:
        print(r)
