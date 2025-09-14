#!/usr/bin/env python3
"""
Run Baseline OCR Model Only - Clean Version
"""

import os
import sys
import time
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

# Import our modules
try:
    sys.path.append('..')
    from main_ocr import SimpleOCRProcessor
    from text_normalizer import TextNormalizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure main_ocr.py is in the parent directory")
    sys.exit(1)

class BaselineOCRRunner:
    def __init__(self):
        self.results = []
        self.start_time = None
        self.normalizer = TextNormalizer()
        self.end_time = None
        
    def run_baseline_ocr(self, image_folder='../images/', ground_truth_file='../data.csv', limit=20):
        """Run baseline OCR on specified number of images"""
        print("="*60)
        print("RUNNING BASELINE OCR MODEL")
        print("="*60)
        
        self.start_time = time.time()
        
        # Load ground truth
        print("Loading ground truth data...")
        gt_df = pd.read_csv(ground_truth_file, sep=';')
        # Clean column names
        gt_df.columns = gt_df.columns.str.strip()
        print(f"Loaded {len(gt_df)} ground truth records")
        
        # Initialize processor
        print("Initializing PaddleOCR...")
        processor = SimpleOCRProcessor()
        processor.set_template('../dat_template.png')
        print("OCR initialized successfully!")
        
        # Process images
        print(f"Processing first {limit} images...")
        results = []
        gt_df_limited = gt_df.head(limit)
        
        for idx, row in gt_df_limited.iterrows():
            image_name = row['Image Name']
            image_path = os.path.join(image_folder, f"{image_name}.jpg")
            
            print(f"Processing {image_name} ({idx+1}/{limit})...")
            
            if os.path.exists(image_path):
                try:
                    result = processor.process_single_image(image_path)
                    result['image_name'] = image_name
                    result['ground_truth'] = row.to_dict()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {image_name}: {e}")
                    results.append({
                        'image_name': image_name,
                        'ground_truth': row.to_dict(),
                        'error': str(e)
                    })
            else:
                print(f"Image not found: {image_path}")
                results.append({
                    'image_name': image_name,
                    'ground_truth': row.to_dict(),
                    'error': 'Image not found'
                })
        
        self.end_time = time.time()
        self.results = results
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy(results, gt_df)
        
        # Save results
        self.save_results(results, accuracy)
        
        print(f"\nBaseline OCR completed in {self.end_time - self.start_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.2f}%")
        
        return results, accuracy
    
    def calculate_accuracy(self, predictions, ground_truth):
        """Calculate accuracy of predictions against ground truth with improved matching"""
        if not predictions or len(predictions) == 0:
            return 0.0
        
        correct = 0
        total = 0
        
        for pred in predictions:
            if 'error' in pred:
                continue
                
            image_name = pred.get('image_name', '')
            gt_row = ground_truth[ground_truth['Image Name'] == image_name]
            
            if len(gt_row) > 0:
                gt_row = gt_row.iloc[0]
                
                # Check key fields with improved matching
                fields_to_check = [
                    ('student_name', 'Student Name'),
                    ('student_id', 'Student ID'), 
                    ('vehicle_plate', 'Vehicle Plate'),
                    ('instructor_name', 'Instructor Name'),
                    ('distance_completed', 'Distance Completed (km)'),
                    ('time_completed', 'Time Completed'),
                    ('distance_remaining', 'Distance Remaining (km)'),
                    ('time_remaining', 'Time Remaining'),
                    ('total_sessions', 'Total Sessions')
                ]
                
                for pred_field, gt_field in fields_to_check:
                    pred_value = str(pred.get(pred_field, '')).strip()
                    gt_value = str(gt_row[gt_field]).strip() if pd.notna(gt_row[gt_field]) else ''
                    
                    # Count all non-empty ground truth values (consistent with save_summary)
                    if gt_value and gt_value != 'nan':
                        total += 1
                        
                        # Normalize predicted value (same as in CSV)
                        pred_norm = self.normalizer.normalize_text(pred_value, field_type=pred_field)
                        
                        # Enhanced matching logic for student_id
                        if pred_field == 'student_id':
                            if self.normalizer.enhanced_student_id_match(pred_norm, gt_value):
                                correct += 1
                        elif pred_field in ['student_name', 'instructor_name']:
                            # Use fuzzy matching for name fields with normalized predicted value
                            if self.normalizer.fuzzy_match(pred_norm, gt_value):
                                correct += 1
                        else:
                            # Normalize ground truth for comparison
                            gt_norm = self.normalizer.normalize_text(gt_value, field_type=pred_field)
                            
                            # Use numeric comparison for better accuracy
                            if is_numeric_equal(pred_norm, gt_norm):
                                correct += 1
        
        return (correct / total * 100) if total > 0 else 0.0
    
    def save_results(self, results, accuracy):
        """Save results to CSV and summary files"""
        # Use fixed filenames - no timestamp
        csv_filename = "baseline_results.csv"
        summary_filename = "baseline_summary.txt"
        
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

def main():
    """Main function to run baseline OCR"""
    runner = BaselineOCRRunner()
    results, accuracy = runner.run_baseline_ocr(limit=100)
    
    print(f"\n🎉 Baseline OCR completed!")
    print(f"📊 Overall Accuracy: {accuracy:.2f}%")
    print(f"📁 Results saved to CSV and summary files")

if __name__ == "__main__":
    main()
