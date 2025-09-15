import cv2
import numpy as np
import easyocr
import os
import csv

class EasyOCRProcessor:
    def __init__(self):
        """Initialize EasyOCR with Vietnamese and English language support"""
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['vi', 'en'], gpu=True)
        print("EasyOCR initialized successfully!")
        
        self.template_image = None
        
        self.regions = {
            'student_name': (40, 440, 380, 500),      # Left panel - student name area
            'student_id': (20, 390, 380, 444),        # Left panel - student ID area  
            'instructor_name': (980, 290, 1250, 400), # Right panel - instructor name
            'vehicle_plate': (65, 600, 310, 670),     # Bottom left - vehicle plate
            'distance_completed': (480, 460, 660, 540), # Center - completed distance
            'time_completed': (500, 520, 660, 600),   # Center - completed time
            'distance_remaining': (650, 460, 900, 540), # Center - remaining distance  
            'time_remaining': (700, 520, 860, 600),   # Center - remaining time
            'total_sessions': (380, 460, 520, 580),    # Center - current speed
        }
        
        self.standard_size = (1280, 720)

    def set_template(self, template_path):
        """Set template image for template matching detection"""
        try:
            self.template_image = cv2.imread(template_path)
            if self.template_image is None:
                raise ValueError(f"Could not load template image: {template_path}")
            
            print(f"Template loaded: {template_path}, size: {self.template_image.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading template: {e}")
            return False

    def detect_dat_interface(self, image):
        """Detect DAT interface using multiple methods like main_ocr.py"""
        print("[DEBUG] Starting DAT interface detection...")
        
        methods = [
            self._detect_by_template_matching,
            self._detect_by_improved_color_analysis,
            self._detect_by_text_regions,
            self._detect_by_adaptive_contours,
            self._detect_by_improved_statistics
        ]
        
        for i, method in enumerate(methods):
            try:
                print(f"[DEBUG] Trying method {i+1}: {method.__name__}")
                result = method(image)
                if result is not None:
                    print(f"[DEBUG] Method {i+1} succeeded!")
                    return result
            except Exception as e:
                print(f"[DEBUG] Method {i+1} failed: {e}")
                continue
        
        print("[DEBUG] All detection methods failed, using full image")
        return image

    def _detect_by_template_matching(self, image):
        """Template matching with multiple scales like main_ocr.py"""
        if self.template_image is None:
            return None
            
        try:
            template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            scales = np.linspace(0.3, 2.0, 20)
            best_match = None
            best_score = 0.3  # Minimum threshold
            
            for scale in scales:
                scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale)
                
                if (scaled_template.shape[0] > image_gray.shape[0] or 
                    scaled_template.shape[1] > image_gray.shape[1]):
                    continue
                
                result = cv2.matchTemplate(image_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    h, w = scaled_template.shape
                    x, y = max_loc
                    best_match = (x, y, w, h)
            
            if best_match:
                x, y, w, h = best_match
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                print(f"[DEBUG] Template matching found region: ({x},{y}) size {w}x{h}, score: {best_score}")
                return image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Template matching failed: {e}")
            return None

    def _detect_by_improved_color_analysis(self, image):
        """Improved color-based detection with adaptive thresholds like main_ocr.py"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            mean_brightness = np.mean(gray)
            print(f"[DEBUG] Image mean brightness: {mean_brightness}")
            
            if mean_brightness > 120:  # Bright image
                lower_light = np.array([0, 0, 120])
                upper_light = np.array([180, 80, 255])
            elif mean_brightness > 80:  # Medium brightness
                lower_light = np.array([0, 0, 80])
                upper_light = np.array([180, 100, 255])
            else:  # Dark image
                lower_light = np.array([0, 0, 40])
                upper_light = np.array([180, 120, 255])
            
            print(f"[DEBUG] Using HSV range: {lower_light} to {upper_light}")
            
            # Create mask for interface area
            mask = cv2.inRange(hsv, lower_light, upper_light)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > (image.shape[0] * image.shape[1] * 0.1):
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Add padding
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    
                    print(f"[DEBUG] Color analysis found region: ({x},{y}) size {w}x{h}")
                    return image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Color analysis failed: {e}")
            return None

    def _detect_by_text_regions(self, image):
        """Detect DAT interface by looking for characteristic text using EasyOCR"""
        try:
            results = self.reader.readtext(image)
            
            if not results:
                return None
            
            dat_keywords = ['DAT', 'PHIÊN HỌC', 'km/h', 'TRUNG TÂM', 'HỌC VIÊN', 'GIÁO VIÊN']
            
            relevant_boxes = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Confidence threshold
                    text_upper = text.upper()
                    for keyword in dat_keywords:
                        if keyword in text_upper:
                            relevant_boxes.append(bbox)
                            break
            
            if len(relevant_boxes) >= 2:  # Need at least 2 DAT-related texts
                # Find bounding box of all relevant text regions
                all_points = np.array([point for bbox in relevant_boxes for point in bbox])
                x_min, y_min = np.min(all_points, axis=0).astype(int)
                x_max, y_max = np.max(all_points, axis=0).astype(int)
                
                # Expand the bounding box
                margin = 50
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(image.shape[1], x_max + margin)
                y_max = min(image.shape[0], y_max + margin)
                
                w, h = x_max - x_min, y_max - y_min
                if w > 200 and h > 150:  # Reasonable size
                    print(f"[DEBUG] Text-based detection: ({x_min}, {y_min}) size {w}x{h}")
                    return image[y_min:y_max, x_min:x_max]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Text-based detection failed: {e}")
            return None

    def _detect_by_adaptive_contours(self, image):
        """Adaptive contour detection like main_ocr.py"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            mean_val = np.mean(gray)
            threshold = max(100, min(200, mean_val + 30))
            
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Too small
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 3.0:
                    coverage = area / (image.shape[0] * image.shape[1])
                    if 0.1 < coverage < 0.9:
                        candidates.append((contour, area, x, y, w, h, coverage))
            
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                _, _, x, y, w, h, _ = candidates[0]
                
                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                return image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Adaptive contours failed: {e}")
            return None

    def _detect_by_improved_statistics(self, image):
        """Improved statistical analysis like main_ocr.py"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            row_means = np.mean(gray, axis=1)
            row_threshold = np.mean(row_means) * 0.4
            
            print(f"[DEBUG] Row threshold: {row_threshold:.1f}, mean: {np.mean(row_means):.1f}")
            
            bright_rows = np.where(row_means > row_threshold)[0]
            if len(bright_rows) == 0:
                return None
            
            top_row = bright_rows[0]
            bottom_row = bright_rows[-1]
            
            col_means = np.mean(gray, axis=0)
            col_threshold = np.mean(col_means) * 0.4
            
            bright_cols = np.where(col_means > col_threshold)[0]
            if len(bright_cols) == 0:
                return None
            
            left_col = bright_cols[0]
            right_col = bright_cols[-1]
            
            new_w = right_col - left_col
            new_h = bottom_row - top_row
            
            if new_w > w * 0.3 and new_h > h * 0.3:
                cropped = image[top_row:bottom_row, left_col:right_col]
                print(f"[DEBUG] Statistical analysis: cropped to ({left_col}, {top_row}) size {new_w}x{new_h}")
                return cropped
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Statistical analysis failed: {e}")
            return None

    def normalize_dat_image(self, dat_interface):
        """Normalize DAT interface to standard size like main_ocr.py"""
        if dat_interface is None:
            return None
        
        return cv2.resize(dat_interface, self.standard_size, interpolation=cv2.INTER_AREA)

    def preprocess_normalized_image(self, image):
        """Preprocessing like main_ocr.py"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        processed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        processed = cv2.GaussianBlur(processed, (1, 1), 0)
        
        return processed

    def extract_text_from_region(self, image, region_name, bbox):
        """Extract text from a specific region using EasyOCR"""
        try:
            x1, y1, x2, y2 = bbox
            
            region_image = image[y1:y2, x1:x2]
            
            if region_image.size == 0:
                print(f"[DEBUG] Empty region for {region_name}")
                return ""
            
            if region_image.shape[0] < 50 or region_image.shape[1] < 100:
                scale_factor = 2
                region_image = cv2.resize(region_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Use EasyOCR
            results = self.reader.readtext(region_image)
            
            print(f"[DEBUG] Region {region_name} EasyOCR result: {results}")
            
            if results:
                texts = [text for (bbox, text, confidence) in results if confidence > 0.5]
                if texts:
                    raw_text = " ".join(texts).strip()
                    return self.clean_ocr_text(raw_text, region_name)
            
            return ""
            
        except Exception as e:
            print(f"[DEBUG] Error extracting from region {region_name}: {e}")
            return ""

    def clean_ocr_text(self, text, region_name):
        """Clean OCR text using same logic as main_ocr.py"""
        if not text:
            return ""
        
        text_lower = text.lower()
        
        prefixes_to_remove = [
            'hv:', 'mahv:', '4v:', 's24', 'loading...', 'hang het ha', 'hang hét c1', 
            'hang', 'hét', 'c1', 'loading', 'ma hv:', 'ma hv', 'hv ', 'mahv ', '4v ',
            'mahv:79335-20250513082447007', 'adng hien trang', 'khoá: k06a c1',
            'khoá:', 'k06a', 'c1', 'hang hét', 'hang het', 'khoa:', 'khoa ',
            'u i khoá:', 'u i khoa:', 'erzura', 'hang hét c1', 'hang het c1'
        ]
        
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                text_lower = text.lower()
            if prefix.lower() in text_lower:
                text = text.replace(prefix, '').strip()
                text_lower = text.lower()
        
        artifacts_to_remove = [
            'hang het ha', 'hang hét c1', 'hang hét', 'hang het',
            'adng hien trang', 'erzura', 'mu4', 'ên', 'uen', 'phiên 7'
        ]
        
        for artifact in artifacts_to_remove:
            if artifact.lower() in text_lower:
                text = text.replace(artifact, '').strip()
                text_lower = text.lower()
        
        text = ' '.join(text.split())
        
        return text.strip()

    def process_single_image(self, image_path):
        """Process a single image using region-based OCR like main_ocr.py"""
        print(f"Processing: {image_path}")
        
        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            dat_interface = self.detect_dat_interface(original_image)
            normalized_dat = self.normalize_dat_image(dat_interface)
            processed_image = self.preprocess_normalized_image(normalized_dat)
            
            extracted_info = {}
            
            for region_name, bbox in self.regions.items():
                print(f"Processing region: {region_name}")
                text = self.extract_text_from_region(processed_image, region_name, bbox)
                extracted_info[region_name] = text
                print(f"  -> Extracted: '{text}'")
                
            return extracted_info
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}

def main():
    print("ssssss")
    # Initialize processor
    # processor = EasyOCRProcessor()
    
    # # Set template
    # template_path = '../dat_template.png'
    # if not processor.set_template(template_path):
    #     print("Warning: Could not load template, proceeding without template matching")
    
    # # Process images
    # image_folder = '../images'
    # ground_truth_file = '../ground_truth.csv'
    # output_file = 'easyocr_conditions_results.csv'
    
    # if not os.path.exists(image_folder):
    #     print(f"Error: Image folder '{image_folder}' not found")
    #     return
    
    # if not os.path.exists(ground_truth_file):
    #     print(f"Error: Ground truth file '{ground_truth_file}' not found")
    #     return
    
    # # Load ground truth
    # ground_truth = {}
    # with open(ground_truth_file, 'r', encoding='utf-8') as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         ground_truth[row['image_name']] = row
    
    # # Process all images
    # results = []
    # image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # for image_file in image_files:
    #     image_path = os.path.join(image_folder, image_file)
    #     extracted_info = processor.process_single_image(image_path)
        
    #     result_row = {'image_name': image_file}
    #     result_row.update(extracted_info)
    #     results.append(result_row)
    
    # # Save results
    # if results:
    #     fieldnames = ['image_name'] + list(processor.regions.keys())
    #     with open(output_file, 'w', newline='', encoding='utf-8') as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         writer.writerows(results)
        
    #     print(f"Results saved to {output_file}")
        
    #     # Calculate accuracy
    #     normalizer = TextNormalizer()
    #     total_fields = 0
    #     correct_fields = 0
        
    #     for result in results:
    #         image_name = result['image_name']
    #         if image_name in ground_truth:
    #             gt_row = ground_truth[image_name]
                
    #             for field in processor.regions.keys():
    #                 if field in gt_row and field in result:
    #                     total_fields += 1
    #                     predicted = normalizer.normalize_text(result[field])
    #                     actual = normalizer.normalize_text(gt_row[field])
                        
    #                     if predicted == actual:
    #                         correct_fields += 1
        
    #     if total_fields > 0:
    #         accuracy = (correct_fields / total_fields) * 100
    #         print(f"Overall accuracy: {accuracy:.2f}% ({correct_fields}/{total_fields})")
    #     else:
    #         print("No fields to evaluate")

if __name__ == "__main__":
    main()
