import os
import time
import cv2
from ultralytics import YOLO
from google.cloud import vision
from accuracy import AccuracyEvaluator

import pandas as pd
import numpy as np

def is_numeric_equal(str1, str2):
    """Check if two strings represent the same numeric value"""
    try:
        # Try to convert to float and compare
        num1 = float(str1)
        num2 = float(str2)
        return abs(num1 - num2) < 1e-9  # Handle floating point precision
    except (ValueError, TypeError):
        # If not numeric, do string comparison
        return str1 == str2
from datetime import datetime


class YOLOGoogleVisionPipeline:
    def __init__(self, yolo_model_path, conf=0.25, processor=None):
        """
        Khởi tạo pipeline với YOLO model, Google Vision client và optional processor
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.client = vision.ImageAnnotatorClient()
        self.conf = conf
        self.names = self.yolo_model.names
        self.processor = processor   # có thể None hoặc 1 class xử lý ảnh

    def run(self, image_path, show=False):
        """
        Chạy pipeline: (optional processor) -> YOLO detect -> crop -> Google Vision OCR -> annotate
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] File {image_path} not found!")
            return None, None

        # Nếu có processor thì xử lý trước
        if self.processor is not None:
            print("[INFO] Running preprocessing with processor...")
            processed_images = self.processor.visualize_processing_steps(image_path)
            if "preprocessed" in processed_images:
                print("[INFO] Using preprocessed image...")
                image = processed_images["preprocessed"]
            else:
                image = cv2.imread(image_path)
        else:
            image = cv2.imread(image_path)

        results = self.yolo_model.predict(image, conf=self.conf, verbose=False)

        outputs = []
        start_time = time.time()

        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0].item())
                class_name = self.names[cls_id]

                # Crop vùng bbox
                crop = image[y1:y2, x1:x2]

                # Encode crop sang bytes cho GCV
                success, encoded_image = cv2.imencode(".jpg", crop)
                content = encoded_image.tobytes()

                gcv_image = vision.Image(content=content)
                response = self.client.text_detection(image=gcv_image)
                texts = response.text_annotations
                text = texts[0].description.strip() if texts else None

                # Vẽ bbox + text
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {text}" if text else class_name
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                outputs.append({
                    "bbox_id": i,
                    "class": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "ocr_text": text
                })

                print(f"[BBox {i}] Class: {class_name} | OCR: {text}")

        end_time = time.time()
        print(f"⏱ Tổng thời gian xử lý: {end_time - start_time:.2f} giây")

        # Nếu có output_path thì lưu ảnh
        # if output_path:
        #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #     cv2.imwrite(output_path, image)
        #     print(f"[INFO] Saved result image to {output_path}")

        if show:
            cv2.imshow("Detections + OCR", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        # Load ground truth
        print("Loading ground truth data...")
        gt_df = pd.read_csv('../data.csv', sep=';')
        # Clean column names
        gt_df.columns = gt_df.columns.str.strip()
        print(f"Loaded {len(gt_df)} ground truth records")
        
       
            
        evaluator = AccuracyEvaluator("../data.csv")
        
        acc = evaluator.calculate_accuracy(outputs, "dat_014")
        print(f"Accuracy: {acc:.2f}%")

        return image, outputs
    
    
    def calculate_accuracy(predictions, image_name, gt_dict):
        """predictions: list text OCR từ model
       image_name: tên ảnh đang test (ví dụ 'dat_000')
       gt_dict: mapping từ csv
    """
        gt_text = str(gt_dict.get(image_name, "")).strip()
        if not gt_text:
            return 0.0
    
    # kiểm tra trùng biển số (so sánh exact match hoặc fuzzy)
        correct = 0
        for pred in predictions:
            if pred.strip() == gt_text:
                correct += 1
                break  # chỉ cần 1 match
    
        return 100.0 if correct > 0 else 0.0

    
    def save_results(self, results, accuracy):
        """Save results to CSV and summary files"""
        # Use fixed filenames - no timestamp
        csv_filename = "gcv_results.csv"
        summary_filename = "gcv_summary.txt"
        
        # Save detailed results CSV
        self.save_detailed_results(results, csv_filename)
        
        # Save summary
        self.save_summary(results, accuracy, summary_filename)
        
        print(f"\nResults saved:")
        print(f"  - Detailed CSV: {csv_filename}")
        print(f"  - Summary: {summary_filename}")
    
    def save_detailed_results(self, results, filename):
        """Save detailed results to CSV"""
        if not results:
            return
        
        # Prepare data for CSV
        csv_data = []
        fields = [
            'student_name', 'student_id', 'vehicle_plate', 'instructor_name',
            'distance_completed', 'time_completed', 'distance_remaining', 
            'time_remaining', 'total_sessions'
        ]
        
        gt_field_mapping = {
            'student_name': 'Student Name',
            'student_id': 'Student ID',
            'vehicle_plate': 'Vehicle Plate',
            'instructor_name': 'Instructor Name',
            'distance_completed': 'Distance Completed (km)',
            'time_completed': 'Time Completed',
            'distance_remaining': 'Distance Remaining (km)',
            'time_remaining': 'Time Remaining',
            'total_sessions': 'Total Sessions'
        }
        
        for result in results:
            if 'error' in result:
                continue
                
            pred_data = {
                'image_name': result.get('image_name', ''),
                'model': 'baseline_ocr',
                'processing_time': 0  # Will be calculated later
            }
            
            gt = result.get('ground_truth', {})
            
            for field in fields:
                # Get raw predicted value
                pred_value = result.get(field, '')
                
                # Normalize predicted value for CSV output
                normalized_pred = self.normalizer.normalize_text(pred_value, field_type=field)
                
                # Store NORMALIZED predicted value in CSV
                pred_data[f'predicted_{field}'] = normalized_pred
                pred_data[f'ground_truth_{field}'] = gt.get(gt_field_mapping[field], '')
            
            csv_data.append(pred_data)
        
        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
            print(f"Saved {len(csv_data)} detailed results to {filename}")
    
    def save_summary(self, results, accuracy, filename):
        """Save summary to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("BASELINE OCR MODEL RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Processing Time: {self.end_time - self.start_time:.2f} seconds\n")
            f.write(f"Total Images Processed: {len(results)}\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
            
            # Field-wise accuracy
            f.write("FIELD-WISE ACCURACY:\n")
            f.write("-" * 30 + "\n")
            
            fields = [
                'student_name', 'student_id', 'vehicle_plate', 'instructor_name',
                'distance_completed', 'time_completed', 'distance_remaining', 
                'time_remaining', 'total_sessions'
            ]
            
            for field in fields:
                field_accuracy = self.calculate_field_accuracy(results, field)
                f.write(f"{field.replace('_', ' ').title()}: {field_accuracy:.2f}%\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*60 + "\n")
    
    def calculate_field_accuracy(self, results, field_name):
        """Calculate accuracy for a specific field"""
        if not results:
            return 0.0
        
        correct = 0
        total = 0
        
        gt_field_mapping = {
            'student_name': 'Student Name',
            'student_id': 'Student ID',
            'vehicle_plate': 'Vehicle Plate',
            'instructor_name': 'Instructor Name',
            'distance_completed': 'Distance Completed (km)',
            'time_completed': 'Time Completed',
            'distance_remaining': 'Distance Remaining (km)',
            'time_remaining': 'Time Remaining',
            'total_sessions': 'Total Sessions'
        }
        
        for result in results:
            if 'error' in result:
                continue
            
            pred_value = str(result.get(field_name, '')).strip()
            gt = result.get('ground_truth', {})
            gt_value = str(gt.get(gt_field_mapping[field_name], '')).strip()
            
            if pred_value and gt_value and pred_value != 'nan' and gt_value != 'nan':
                total += 1
                
                # Enhanced matching logic for student_id
                if field_name == 'student_id':
                    if self.normalizer.enhanced_student_id_match(pred_value, gt_value):
                        correct += 1
                else:
                    # Normalize values for comparison
                    pred_norm = self.normalizer.normalize_text(pred_value, field_type=field_name)
                    gt_norm = self.normalizer.normalize_text(gt_value, field_type=field_name)
                    
                    # STRICT EXACT MATCHING ONLY
                    if pred_norm == gt_norm:
                        correct += 1
        
        return (correct / total * 100) if total > 0 else 0.0