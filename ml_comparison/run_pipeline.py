#!/usr/bin/env python3
"""
Text Correction ML Pipeline
Pipeline that runs Baseline OCR + Text Correction ML
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

# Import our modules
sys.path.append('..')
from run_baseline import BaselineOCRRunner
from text_correction_ml import SimpleTextCorrectionML
from text_normalizer import TextNormalizer

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

class TextCorrectionPipeline:
    def __init__(self):
        self.baseline_runner = BaselineOCRRunner()
        self.correction_ml = SimpleTextCorrectionML()
        self.normalizer = TextNormalizer()
        
    def load_ground_truth(self, csv_path="../data.csv"):
        """Load ground truth data from CSV"""
        try:
            df = pd.read_csv(csv_path, sep=';')
            gt_data = {}
            
            for _, row in df.iterrows():
                image_name = str(row['Image Name']).replace('.jpg', '').strip()
                gt_data[image_name] = {
                    'student_name': str(row.get('Student Name', '')).strip(),
                    'student_id': str(row.get('Student ID', '')).strip(),
                    'vehicle_plate': str(row.get('Vehicle Plate', '')).strip(),
                    'instructor_name': str(row.get('Instructor Name', '')).strip(),
                    'distance_completed': str(row.get('Distance Completed (km)', '')).strip(),
                    'time_completed': str(row.get('Time Completed', '')).strip(),
                    'distance_remaining': str(row.get('Distance Remaining (km)', '')).strip(),
                    'time_remaining': str(row.get('Time Remaining', '')).strip(),
                    'total_sessions': str(row.get('Total Sessions', '')).strip()
                }
            
            print(f"📊 Loaded ground truth for {len(gt_data)} images")
            return gt_data
            
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def run_baseline_phase(self, images_dir="../images", gt_data=None):
        """Phase 1: Run Baseline OCR"""
        print("\n" + "=" * 60)
        print("🚀 PHASE 1: BASELINE OCR")
        print("=" * 60)
        
        print("📸 Running Baseline OCR...")
        
        # Run baseline OCR - process all 100 images
        self.baseline_runner.run_baseline_ocr(
            image_folder=images_dir + '/',
            ground_truth_file='../data.csv',
            limit=100  # Process all 100 images
        )
        
        # Convert results
        baseline_results = []
        for result in self.baseline_runner.results:
            gt_raw = result['ground_truth']
            gt_converted = {
                'student_name': str(gt_raw.get('Student Name', '')).strip(),
                'student_id': str(gt_raw.get('Student ID', '')).strip(),
                'vehicle_plate': str(gt_raw.get('Vehicle Plate', '')).strip(),
                'instructor_name': str(gt_raw.get('Instructor Name', '')).strip(),
                'distance_completed': str(gt_raw.get('Distance Completed (km)', '')).strip(),
                'time_completed': str(gt_raw.get('Time Completed', '')).strip(),
                'distance_remaining': str(gt_raw.get('Distance Remaining (km)', '')).strip(),
                'time_remaining': str(gt_raw.get('Time Remaining', '')).strip(),
                'total_sessions': str(gt_raw.get('Total Sessions', '')).strip()
            }
            
            converted_result = {
                'image_name': result['image_name'],
                'processing_time': 0,
                'ground_truth': gt_converted
            }
            
            # Add predicted fields
            for field in ['student_name', 'student_id', 'vehicle_plate', 'instructor_name',
                         'distance_completed', 'time_completed', 'distance_remaining',
                         'time_remaining', 'total_sessions']:
                converted_result[field] = result.get(field, '')
            
            baseline_results.append(converted_result)
        
        print(f"\n🎉 Baseline phase completed")
        print(f"📊 Processed {len(baseline_results)} images")
        
        return baseline_results
    
    def run_correction_phase(self, baseline_results):
        """Phase 2: Apply Text Correction ML"""
        print("\n" + "=" * 60)
        print("🧠 PHASE 2: TEXT CORRECTION ML")
        print("=" * 60)
        
        print("🤖 Training Text Correction ML...")
        
        # Train correction model
        training_data = self.correction_ml.create_training_data()
        if training_data:
            self.correction_ml.train(training_data)
            # Save trained model
            self.correction_ml.save_model()
        else:
            print("❌ No training data available")
            return baseline_results
        
        print("🔧 Applying text correction to baseline results...")
        
        # Apply correction
        corrected_results = self.correction_ml.correct_baseline_results(baseline_results)
        
        print(f"\n🎉 Correction phase completed")
        print(f"📊 Corrected {len(corrected_results)} results")
        
        return corrected_results
    
    def calculate_accuracy(self, results, gt_data=None):
        """Calculate accuracy of results"""
        if not results:
            return 0.0
        
        correct = 0
        total = 0
        
        for result in results:
            if 'error' in result:
                continue
                
            gt = result.get('ground_truth', {})
            
            # Check key fields
            fields_to_check = [
                'student_name', 'student_id', 'vehicle_plate', 'instructor_name',
                'distance_completed', 'time_completed', 'distance_remaining', 
                'time_remaining', 'total_sessions'
            ]
            
            for field in fields_to_check:
                pred_value = result.get(field, '')
                gt_value = gt.get(field, '')
                
                if pred_value and gt_value and pred_value != 'nan' and gt_value != 'nan':
                    total += 1
                    
                    # Enhanced matching logic for student_id
                    if field == 'student_id':
                        if self.normalizer.enhanced_student_id_match(pred_value, gt_value):
                            correct += 1
                    else:
                        # Normalize values for comparison
                        pred_norm = self.normalizer.normalize_text(pred_value, field_type=field)
                        gt_norm = self.normalizer.normalize_text(gt_value, field_type=field)
                        
                        # STRICT EXACT MATCHING ONLY
                        if pred_norm == gt_norm:
                            correct += 1
        
        return (correct / total * 100) if total > 0 else 0.0
    
    def save_results(self, results, accuracy, processing_time, model_name):
        """Save results to CSV and summary"""
        if not results:
            print("❌ No results to save")
            return
        
        # Fixed filenames - overwrite each time
        if "baseline" in model_name.lower():
            csv_filename = "baseline_results.csv"
            summary_filename = "baseline_summary.txt"
        else:
            csv_filename = "enhanced_results.csv"
            summary_filename = "enhanced_summary.txt"
        
        # Save detailed CSV
        self.save_csv(results, csv_filename)
        
        # Save summary
        self.save_summary(results, accuracy, processing_time, summary_filename)
        
        print(f"\n📁 Results saved:")
        print(f"  - Detailed CSV: {csv_filename}")
        print(f"  - Summary: {summary_filename}")
        print(f"  - Accuracy: {accuracy:.2f}%")
    
    def save_csv(self, results, filename):
        """Save results to CSV"""
        csv_data = []
        fields = [
            'student_name', 'student_id', 'vehicle_plate', 'instructor_name',
            'distance_completed', 'time_completed', 'distance_remaining', 
            'time_remaining', 'total_sessions'
        ]
        
        for result in results:
            if 'error' in result:
                continue
                
            pred_data = {
                'image_name': result.get('image_name', ''),
                'model': 'text_correction_ml',
                'processing_time': 0
            }
            
            gt = result.get('ground_truth', {})
            
            for field in fields:
                # Get predicted value
                pred_value = result.get(field, '')
                
                # Store predicted value in CSV
                pred_data[f'predicted_{field}'] = pred_value
                pred_data[f'ground_truth_{field}'] = gt.get(field, '')
            
            csv_data.append(pred_data)
        
        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
    
    def save_summary(self, results, accuracy, processing_time, filename):
        """Save detailed summary to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("OCR MODEL RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Processing Time: {processing_time:.2f} seconds\n")
            f.write(f"Total Images Processed: {len(results)}\n")
            
            # Calculate field-specific accuracy from CSV file
            csv_filename = filename.replace('_summary.txt', '_results.csv')
            if os.path.exists(csv_filename):
                df = pd.read_csv(csv_filename)
                fields = ['student_name', 'student_id', 'vehicle_plate', 'instructor_name',
                         'distance_completed', 'time_completed', 'distance_remaining',
                         'time_remaining', 'total_sessions']
                
                # Calculate overall accuracy from field-specific
                total_correct = 0
                total_fields = 0
                
                f.write("FIELD-SPECIFIC ACCURACY:\n")
                f.write("-" * 40 + "\n")
                
                for field in fields:
                    correct = 0
                    total = 0
                    
                    for _, row in df.iterrows():
                        predicted = str(row.get(f'predicted_{field}', '')).strip()
                        ground_truth = str(row.get(f'ground_truth_{field}', '')).strip()
                        
                        if ground_truth and ground_truth != 'nan':
                            total += 1
                            
                            # Use fuzzy matching for name fields
                            if field in ['student_name', 'instructor_name']:
                                from text_normalizer import TextNormalizer
                                normalizer = TextNormalizer()
                                if normalizer.fuzzy_match(predicted, ground_truth):
                                    correct += 1
                            else:
                                # Use numeric comparison for other fields
                                if is_numeric_equal(predicted, ground_truth):
                                    correct += 1
                    
                    total_correct += correct
                    total_fields += total
                    accuracy_pct = (correct / total * 100) if total > 0 else 0.0
                    f.write(f"{field.replace('_', ' ').title():<20}: {correct:2d}/{total:2d} ({accuracy_pct:5.1f}%)\n")
                
                # Calculate and display overall accuracy
                overall_accuracy = (total_correct / total_fields * 100) if total_fields > 0 else 0.0
                f.write(f"\nOverall Accuracy: {overall_accuracy:.2f}%\n")
            else:
                f.write("FIELD-SPECIFIC ACCURACY:\n")
                f.write("-" * 40 + "\n")
                f.write("CSV file not found for detailed accuracy calculation\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")

def main():
    """Main execution function"""
    print("🚀 Starting Text Correction ML Pipeline...")
    
    pipeline = TextCorrectionPipeline()
    
    # Load ground truth
    gt_data = pipeline.load_ground_truth()
    
    # Phase 1: Baseline OCR
    start_time = time.time()
    baseline_results = pipeline.run_baseline_phase(gt_data=gt_data)
    baseline_time = time.time() - start_time
    
    # Calculate baseline accuracy
    baseline_accuracy = pipeline.calculate_accuracy(baseline_results, gt_data)
    
    # Save baseline results
    pipeline.save_results(baseline_results, baseline_accuracy, baseline_time, "baseline_ocr")
    
    # Phase 2: Text Correction ML
    start_time = time.time()
    corrected_results = pipeline.run_correction_phase(baseline_results)
    correction_time = time.time() - start_time
    
    # Calculate corrected accuracy
    corrected_accuracy = pipeline.calculate_accuracy(corrected_results, gt_data)
    
    # Save corrected results
    pipeline.save_results(corrected_results, corrected_accuracy, correction_time, "text_correction_ml")
    
    print("\n" + "=" * 60)
    print("🎯 PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"📊 Baseline OCR: {baseline_accuracy:.2f}% accuracy")
    print(f"🧠 Text Correction ML: {corrected_accuracy:.2f}% accuracy")
    print(f"📈 Improvement: {corrected_accuracy - baseline_accuracy:+.2f}%")
    print("📁 Check generated CSV and summary files for detailed results")
    print("=" * 60)

if __name__ == "__main__":
    main()
