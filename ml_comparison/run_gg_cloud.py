from main_gcv import YOLOGoogleVisionPipeline
from main_ocr import SimpleOCRProcessor

if __name__ == "__main__":
    processor = SimpleOCRProcessor()
    test_image = "../test_img/dat_014.jpg"

    # Khởi tạo pipeline với YOLO model
    pipeline = YOLOGoogleVisionPipeline(yolo_model_path="../models/text_detector.pt", conf=0.25, processor=processor)

    print("Running YOLO + Google Vision OCR pipeline...")
    final_image, results = pipeline.run(test_image, show=False)

    if results:
        print("\nProcessing completed successfully!")
        print("Kết quả OCR:")
        for r in results:
            print(r)
    else:
        print("No detections found!")
