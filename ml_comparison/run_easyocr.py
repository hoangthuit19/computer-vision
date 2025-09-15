import os
import sys
import time
import pandas as pd
import numpy as np
import re
import cv2
from PIL import Image, ImageEnhance, ImageFilter

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
    from main_easy_ocr import EasyOCRProcessor
    from text_normalizer import TextNormalizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure text_normalizer.py is in the parent directory")
    sys.exit(1)

class VietnameseDiacriticsEnhancer:
    """Enhanced Vietnamese diacritics processing for better OCR accuracy"""
    
    def __init__(self):
        # Common Vietnamese names and words for context-based correction
        self.vietnamese_names = {
            'nguyen': 'Nguyễn', 'tran': 'Trần', 'le': 'Lê', 'pham': 'Phạm', 
            'hoang': 'Hoàng', 'huynh': 'Huỳnh', 'vo': 'Võ', 'vu': 'Vũ',
            'dang': 'Đặng', 'bui': 'Bùi', 'do': 'Đỗ', 'ho': 'Hồ',
            'ngo': 'Ngô', 'duong': 'Dương', 'ly': 'Lý', 'mai': 'Mai',
            'anh': 'Anh', 'minh': 'Minh', 'duc': 'Đức', 'huy': 'Huy',
            'nam': 'Nam', 'hung': 'Hùng', 'dung': 'Dũng', 'son': 'Sơn',
            'tuan': 'Tuấn', 'hai': 'Hải', 'long': 'Long', 'quan': 'Quân',
            'thang': 'Thắng', 'cuong': 'Cường', 'khang': 'Khang',
            'linh': 'Linh', 'huong': 'Hương', 'lan': 'Lan', 'nga': 'Nga',
            'yen': 'Yến', 'thu': 'Thu', 'ha': 'Hà', 'hong': 'Hồng',
            'thuy': 'Thúy', 'loan': 'Loan', 'lien': 'Liên', 'my': 'Mỹ',
            'thanh': 'Thành', 'van': 'Văn', 'thi': 'Thị', 'xuan': 'Xuân'
        }
        
        # OCR common misreads for Vietnamese characters
        self.diacritic_corrections = {
            # Common OCR errors for Vietnamese diacritics
            'a': ['à', 'á', 'ả', 'ã', 'ạ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
            'e': ['è', 'é', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
            'i': ['ì', 'í', 'ỉ', 'ĩ', 'ị'],
            'o': ['ò', 'ó', 'ỏ', 'õ', 'ọ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
            'u': ['ù', 'ú', 'ủ', 'ũ', 'ụ', 'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
            'y': ['ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'],
            'd': ['đ']
        }
        
        # Pattern-based diacritic restoration
        self.name_patterns = {
            r'\bnguy[e]?n\b': 'Nguyễn',
            r'\btran\b': 'Trần', 
            r'\ble\b': 'Lê',
            r'\bpham\b': 'Phạm',
            r'\bhoang\b': 'Hoàng',
            r'\bdo\b': 'Đỗ',
            r'\bvu\b': 'Vũ',
            r'\bvo\b': 'Võ',
            r'\bdang\b': 'Đặng',
            r'\bminh\b': 'Minh',
            r'\bduc\b': 'Đức',
            r'\btuan\b': 'Tuấn',
            r'\bthang\b': 'Thắng',
            r'\bcuong\b': 'Cường',
            r'\bhuong\b': 'Hương',
            r'\bthanh\b': 'Thành',
            r'\bxuan\b': 'Xuân'
        }
    
    def enhance_image_for_ocr(self, image_path):
        """Enhance image quality for better Vietnamese OCR"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # Convert to PIL for enhancement
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.1)
            
            # Apply slight blur to reduce noise
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Convert back to OpenCV format
            enhanced_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Save enhanced image temporarily
            enhanced_path = image_path.replace('.jpg', '_enhanced.jpg')
            cv2.imwrite(enhanced_path, enhanced_img)
            
            return enhanced_path
            
        except Exception as e:
            print(f"[DEBUG] Image enhancement failed: {e}")
            return image_path
    
    def restore_vietnamese_diacritics(self, text):
        """Restore Vietnamese diacritics using pattern matching and context"""
        if not text:
            return text
        
        # Convert to lowercase for processing
        text_lower = text.lower()
        restored_text = text
        
        # Apply name pattern corrections
        for pattern, replacement in self.name_patterns.items():
            restored_text = re.sub(pattern, replacement, restored_text, flags=re.IGNORECASE)
        
        # Word-by-word restoration for common names
        words = restored_text.split()
        restored_words = []
        
        for word in words:
            word_lower = word.lower()
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word_lower)
            
            if clean_word in self.vietnamese_names:
                # Preserve original case pattern
                if word.isupper():
                    restored_words.append(self.vietnamese_names[clean_word].upper())
                elif word.istitle():
                    restored_words.append(self.vietnamese_names[clean_word])
                else:
                    restored_words.append(self.vietnamese_names[clean_word].lower())
            else:
                restored_words.append(word)
        
        return ' '.join(restored_words)
    
    def post_process_vietnamese_text(self, text):
        """Post-process OCR text to fix Vietnamese-specific issues"""
        if not text:
            return text
        
        # Fix common OCR character substitutions
        corrections = {
            # Numbers often misread as letters in Vietnamese names
            '0': 'o', '1': 'i', '5': 's', '8': 'b', '9': 'g', '6': 'g',
            '2': 'z', '3': 'e', '4': 'a', '7': 't',
            # Common letter confusions
            'rn': 'm', 'cl': 'd', 'ri': 'n'
        }
        
        processed_text = text
        for wrong, correct in corrections.items():
            processed_text = processed_text.replace(wrong, correct)
        
        # Restore diacritics
        processed_text = self.restore_vietnamese_diacritics(processed_text)
        
        return processed_text

# Define SimpleEasyOCRProcessor using EasyOCR
class SimpleEasyOCRProcessor:
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['vi', 'en'], gpu=False)  # Use GPU if available, change to True
        self.vietnamese_enhancer = VietnameseDiacriticsEnhancer()

    def set_template(self, template_path):
        # For compatibility, but not used in this implementation
        print(f"Setting template (not used in EasyOCR impl): {template_path}")

    def process_single_image(self, image_path):
        enhanced_image_path = self.vietnamese_enhancer.enhance_image_for_ocr(image_path)
        
        # Perform EasyOCR with enhanced image
        ocr_results = self.reader.readtext(enhanced_image_path)
        
        # Clean up enhanced image if it was created
        if enhanced_image_path != image_path and os.path.exists(enhanced_image_path):
            try:
                os.remove(enhanced_image_path)
            except:
                pass
        
        print(f"[DEBUG] Raw OCR results for {image_path}: {len(ocr_results)} detections")
        for i, (box, text, conf) in enumerate(ocr_results):
            print(f"  [{i}] Text: '{text}' (confidence: {conf:.2f})")

        enhanced_ocr_results = []
        for box, text, conf in ocr_results:
            # Apply Vietnamese text enhancement
            enhanced_text = self.vietnamese_enhancer.post_process_vietnamese_text(text)
            enhanced_ocr_results.append((box, enhanced_text, conf))
            if enhanced_text != text:
                print(f"  [ENHANCED] '{text}' -> '{enhanced_text}'")

        # Group into lines
        def group_into_lines(results):
            if not results:
                return []
            detections = []
            for box, text, conf in results:
                avg_y = sum(point[1] for point in box) / 4
                avg_x = sum(point[0] for point in box) / 4
                detections.append((avg_y, avg_x, text, box, conf))
            # Sort by y, then x
            detections.sort(key=lambda x: (x[0], x[1]))
            lines = []
            current_line = []
            current_y = detections[0][0] if detections else 0
            for det in detections:
                y = det[0]
                if abs(y - current_y) > 30:  # Increased threshold for new line
                    if current_line:
                        lines.append(current_line)
                    current_line = []
                    current_y = y
                current_line.append(det)
            if current_line:
                lines.append(current_line)
            # For each line, sort by x and join text
            line_texts = []
            for line in lines:
                line.sort(key=lambda x: x[1])  # sort by avg_x
                line_text = ' '.join(d[2] for d in line)
                line_texts.append(line_text)
            return line_texts

        line_texts = group_into_lines(enhanced_ocr_results)
        
        print(f"[DEBUG] Grouped lines: {len(line_texts)}")
        for i, line in enumerate(line_texts):
            print(f"  Line {i}: '{line}'")

        # Extract fields based on labels and patterns
        extracted = {}
        
        label_patterns = {
            'student_name': [
                r'(?:student\s+name|họ\s+và\s+tên|tên\s+học\s+viên|học\s+viên)[:\s]*(.+)',
                r'(?:hv|mahv)[:\s]*([^0-9\-]+)',  # Look for HV: followed by name (not numbers)
                r'(?:tên|họ\s+tên)[:\s]*([A-Za-zÀ-ỹĐđ\s]+)',  # Vietnamese name pattern
                r'([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+(?:\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+){1,3})',  # Vietnamese name with diacritics
            ],
            'student_id': [
                r'(?:student\s+id|mã\s+học\s+viên|mahv)[:\s]*([0-9\-]+)',
                r'(?:hv|mahv)[:\s]*([0-9\-]{10,})',  # Look for long number sequences
                r'([0-9]{5}\-[0-9]{14,})',  # Pattern like 79035-20250513082029960
            ],
            'vehicle_plate': [
                r'(?:vehicle\s+plate|biển\s+số|bs)[:\s]*([0-9]{2}[A-Z][0-9]{5})',
                r'([0-9]{2}[A-Z][0-9]{5})',  # Direct pattern match
            ],
            'instructor_name': [
                r'(?:instructor|giáo\s+viên|gv)[:\s]*(.+)',
                r'(?:gv)[:\s]*([^0-9\-]+)',
                r'(?:giáo\s+viên)[:\s]*([A-Za-zÀ-ỹĐđ\s]+)',  # Vietnamese instructor name
            ],
            'distance_completed': [
                r'(?:distance\s+completed|quãng\s+đường\s+đã\s+học)[:\s]*([0-9\.]+)',
                r'([0-9]+\.[0-9]+)\s*km',  # Look for decimal numbers followed by km
            ],
            'time_completed': [
                r'(?:time\s+completed|thời\s+gian\s+đã\s+học)[:\s]*([0-9:]+)',
                r'([0-9]{1,2}:[0-9]{2})',  # Time pattern HH:MM
            ],
            'distance_remaining': [
                r'(?:distance\s+remaining|quãng\s+đường\s+còn\s+lại)[:\s]*([0-9\.]+)',
                r'([0-9]+\.[0-9]+)\s*km',  # Look for decimal numbers followed by km
            ],
            'time_remaining': [
                r'(?:time\s+remaining|thời\s+gian\s+còn\s+lại)[:\s]*([0-9:]+)',
                r'([0-9]{1,2}:[0-9]{2})',  # Time pattern HH:MM
            ],
            'total_sessions': [
                r'(?:total\s+sessions|tổng\s+số\s+phiên|phiên)[:\s]*([0-9]+)',
                r'phiên\s+([0-9]+)',
                r'([0-9]+)\s*phiên',
            ]
        }

        # Try to extract each field using patterns
        all_text = ' '.join(line_texts).lower()
        print(f"[DEBUG] Combined text for pattern matching: '{all_text}'")
        
        for field, patterns in label_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                if matches:
                    # Take the first match and clean it
                    value = matches[0].strip()
                    if value:
                        if field in ['student_name', 'instructor_name']:
                            value = self.vietnamese_enhancer.restore_vietnamese_diacritics(value)
                        extracted[field] = value
                        print(f"[DEBUG] Found {field}: '{value}' using pattern: {pattern}")
                        break
            
            # If no pattern match found, try positional extraction for some fields
            if field not in extracted:
                if field == 'student_id':
                    # Look for long number sequences that look like student IDs
                    for line in line_texts:
                        id_matches = re.findall(r'([0-9]{5}\-[0-9]{14,})', line)
                        if id_matches:
                            extracted[field] = id_matches[0]
                            print(f"[DEBUG] Found {field} by position: '{id_matches[0]}'")
                            break
                elif field == 'vehicle_plate':
                    # Look for vehicle plate patterns
                    for line in line_texts:
                        plate_matches = re.findall(r'([0-9]{2}[A-Z][0-9]{5})', line.upper())
                        if plate_matches:
                            extracted[field] = plate_matches[0]
                            print(f"[DEBUG] Found {field} by position: '{plate_matches[0]}'")
                            break
                elif field in ['student_name', 'instructor_name']:
                    for line in line_texts:
                        # Look for Vietnamese name patterns
                        name_matches = re.findall(r'([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+(?:\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+){1,3})', line)
                        if name_matches:
                            name = self.vietnamese_enhancer.restore_vietnamese_diacritics(name_matches[0])
                            extracted[field] = name
                            print(f"[DEBUG] Found {field} by Vietnamese pattern: '{name}'")
                            break

        print(f"[DEBUG] Final extracted data: {extracted}")
        return extracted

class BaselineEasyOCRRunner:
    def __init__(self):
        self.results = []
        self.start_time = None
        self.normalizer = TextNormalizer()
        self.end_time = None
        
    def run_baseline_easy_ocr(self, image_folder='../images/', ground_truth_file='../data.csv', limit=20):
        """Run baseline EasyOCR on specified number of images"""
        print("="*60)
        print("RUNNING ENHANCED EasyOCR MODEL WITH VIETNAMESE DIACRITICS SUPPORT")
        print("="*60)
        
        self.start_time = time.time()
        
        # Load ground truth
        print("Loading ground truth data...")
        gt_df = pd.read_csv(ground_truth_file, sep=';')
        # Clean column names
        gt_df.columns = gt_df.columns.str.strip()
        print(f"Loaded {len(gt_df)} ground truth records")
        
        # Initialize processor
        print("Initializing Enhanced EasyOCR with Vietnamese support...")
        processor = EasyOCRProcessor()
        processor.set_template('../dat_template.png')
        print("Enhanced EasyOCR initialized successfully!")
        
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
        
        print(f"\nEnhanced EasyOCR completed in {self.end_time - self.start_time:.2f} seconds")
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
        csv_filename = "easyocr_results.csv"
        summary_filename = "easyocr_summary.txt"
        
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
                'model': 'easyocr',
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
            f.write("ENHANCED EASYOCR MODEL RESULTS WITH VIETNAMESE DIACRITICS SUPPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Processing Time: {self.end_time - self.start_time:.2f} seconds\n")
            f.write(f"Total Images Processed: {len(results)}\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
            
            f.write("ENHANCEMENTS APPLIED:\n")
            f.write("- Vietnamese diacritics restoration\n")
            f.write("- Enhanced image preprocessing\n")
            f.write("- Context-based name correction\n")
            f.write("- Advanced pattern matching for Vietnamese text\n\n")
            
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
            f.write("END OF ENHANCED REPORT\n")
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
    """Main function to run enhanced EasyOCR with Vietnamese support"""
    runner = BaselineEasyOCRRunner()
    results, accuracy = runner.run_baseline_easy_ocr(limit=100)
    
    print(f"\n🎉 Enhanced EasyOCR with Vietnamese Diacritics Support completed!")
    print(f"📊 Overall Accuracy: {accuracy:.2f}%")
    print(f"📁 Results saved to CSV and summary files")
    print(f"🇻🇳 Vietnamese diacritics restoration applied!")

if __name__ == "__main__":
    main()
