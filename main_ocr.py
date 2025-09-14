import cv2
import numpy as np
from paddleocr import PaddleOCR
import pandas as pd
import re
import os
from pathlib import Path

class SimpleOCRProcessor:
    def __init__(self):
        """Initialize PaddleOCR with Vietnamese language support"""
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi')
        print("OCR initialized successfully!")
        
        self.template_image = None
        self.template_features = None
        self.feature_detector = None
        
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
            
            # Initialize feature detector for advanced matching
            try:
                self.feature_detector = cv2.SIFT_create()
            except:
                try:
                    self.feature_detector = cv2.ORB_create(nfeatures=1000)
                except:
                    print("[WARNING] No feature detector available, using basic template matching only")
                    self.feature_detector = None
            
            # Extract features from template if detector is available
            if self.feature_detector is not None:
                template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
                kp, desc = self.feature_detector.detectAndCompute(template_gray, None)
                self.template_features = (kp, desc)
                print(f"Extracted {len(kp)} features from template")
            
            return True
            
        except Exception as e:
            print(f"Error loading template: {e}")
            self.template_image = None
            self.template_features = None
            return False

    def visualize_processing_steps(self, image_path, save_dir="debug_images"):
        """Visualize and save each processing step for debugging"""
        print(f"Visualizing processing steps for: {image_path}")
        
        # Create debug directory
        os.makedirs(save_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        try:
            # Step 1: Load original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            cv2.imwrite(f"{save_dir}/{base_name}_01_original.jpg", original_image)
            print(f"Saved: {base_name}_01_original.jpg")
            
            # Step 2: Detect and crop DAT interface
            dat_interface = self.detect_dat_interface(original_image)
            cv2.imwrite(f"{save_dir}/{base_name}_02_dat_cropped.jpg", dat_interface)
            print(f"Saved: {base_name}_02_dat_cropped.jpg")
            
            # Step 3: Normalize to standard size
            normalized_dat = self.normalize_dat_image(dat_interface)
            cv2.imwrite(f"{save_dir}/{base_name}_03_normalized.jpg", normalized_dat)
            print(f"Saved: {base_name}_03_normalized.jpg")
            
            # Step 4: Preprocess for OCR
            processed_image = self.preprocess_normalized_image(normalized_dat)
            cv2.imwrite(f"{save_dir}/{base_name}_04_preprocessed.jpg", processed_image)
            print(f"Saved: {base_name}_04_preprocessed.jpg")
            
            # Step 5: Draw regions on processed image
            regions_image = processed_image.copy()
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0)]
            
            for i, (region_name, bbox) in enumerate(self.regions.items()):
                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]
                cv2.rectangle(regions_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(regions_image, region_name, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imwrite(f"{save_dir}/{base_name}_05_regions_marked.jpg", regions_image)
            print(f"Saved: {base_name}_05_regions_marked.jpg")
            
            print(f"All visualization images saved in: {save_dir}/")
            return {
                'original': original_image,
                'dat_cropped': dat_interface,
                'normalized': normalized_dat,
                'preprocessed': processed_image,
                'regions_marked': regions_image
            }
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def detect_dat_interface(self, image):
        """Improved DAT interface detection with multiple strategies"""
        try:
            print(f"[DEBUG] Starting DAT detection on image size: {image.shape}")
            
            # Strategy 1: Template-based detection using known DAT elements
            dat_interface = self._detect_by_template_matching(image)
            if dat_interface is not None:
                print("[DEBUG] Successfully detected using template matching")
                return self._remove_black_borders(dat_interface)
            
            # Strategy 2: Improved color-based detection
            dat_interface = self._detect_by_improved_color_analysis(image)
            if dat_interface is not None:
                print("[DEBUG] Successfully detected using improved color analysis")
                return self._remove_black_borders(dat_interface)
            
            # Strategy 3: Text-based detection (look for DAT-specific text)
            dat_interface = self._detect_by_text_regions(image)
            if dat_interface is not None:
                print("[DEBUG] Successfully detected using text regions")
                return self._remove_black_borders(dat_interface)
            
            # Strategy 4: Contour-based detection with adaptive parameters
            dat_interface = self._detect_by_adaptive_contours(image)
            if dat_interface is not None:
                print("[DEBUG] Successfully detected using adaptive contours")
                return self._remove_black_borders(dat_interface)
            
            print("[DEBUG] All detection strategies failed, using smart crop")
            fallback_result = self._smart_fallback_crop(image)
            return self._remove_black_borders(fallback_result)
                
        except Exception as e:
            print(f"[DEBUG] Error in DAT detection: {e}")
            import traceback
            traceback.print_exc()
            return image

    def _detect_by_template_matching(self, image):
        """Enhanced template matching using actual template image"""
        try:
            # If no template is set, fall back to original method
            if self.template_image is None:
                return self._detect_by_ui_elements(image)
            
            print("[DEBUG] Using template matching with loaded template")
            
            # Method 1: Direct template matching with multiple scales
            result = self._match_template_multiscale(image)
            if result is not None:
                print("[DEBUG] Template matching successful with multiscale")
                return result
            
            # Method 2: Feature-based matching (if available)
            if self.feature_detector is not None and self.template_features is not None:
                result = self._match_template_features(image)
                if result is not None:
                    print("[DEBUG] Template matching successful with features")
                    return result
            
            # Method 3: Structural similarity
            result = self._match_template_structure(image)
            if result is not None:
                print("[DEBUG] Template matching successful with structure")
                return result
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Template matching failed: {e}")
            return None

    def _match_template_multiscale(self, image):
        """Template matching with multiple scales"""
        try:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
            
            best_match = None
            best_score = 0
            
            # Try different scales
            scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
            
            for scale in scales:
                # Resize template
                template_h, template_w = template_gray.shape
                new_w = int(template_w * scale)
                new_h = int(template_h * scale)
                
                if new_w > image_gray.shape[1] or new_h > image_gray.shape[0]:
                    continue
                
                scaled_template = cv2.resize(template_gray, (new_w, new_h))
                
                # Template matching
                result = cv2.matchTemplate(image_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = (max_loc[0], max_loc[1], new_w, new_h, scale)
            
            # If we found a good match
            if best_match is not None and best_score > 0.4:  # Threshold for template matching
                x, y, w, h, scale = best_match
                
                # Add some padding
                padding = int(min(w, h) * 0.02)  # 2% padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                print(f"[DEBUG] Template match found: score={best_score:.3f}, scale={scale:.2f}, pos=({x},{y}), size={w}x{h}")
                return image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Multiscale template matching failed: {e}")
            return None

    def _match_template_features(self, image):
        """Feature-based template matching using SIFT/ORB"""
        try:
            if self.template_features is None:
                return None
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template_kp, template_desc = self.template_features
            
            # Extract features from image
            image_kp, image_desc = self.feature_detector.detectAndCompute(image_gray, None)
            
            if image_desc is None or len(image_desc) < 10:
                return None
            
            # Match features
            if hasattr(cv2, 'BFMatcher'):
                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(template_desc, image_desc, k=2)
                
                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
            else:
                # Fallback to FLANN matcher
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(template_desc, image_desc, k=2)
                
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
            
            print(f"[DEBUG] Found {len(good_matches)} good feature matches")
            
            # Need at least 10 good matches
            if len(good_matches) >= 10:
                # Extract matched points
                template_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                image_pts = np.float32([image_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                homography, mask = cv2.findHomography(template_pts, image_pts, cv2.RANSAC, 5.0)
                
                if homography is not None:
                    # Get template corners
                    h, w = self.template_image.shape[:2]
                    template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    
                    # Transform corners to image space
                    image_corners = cv2.perspectiveTransform(template_corners, homography)
                    
                    # Get bounding box
                    x_coords = image_corners[:, 0, 0]
                    y_coords = image_corners[:, 0, 1]
                    
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    
                    # Validate bounding box
                    if (x_min >= 0 and y_min >= 0 and 
                        x_max <= image.shape[1] and y_max <= image.shape[0] and
                        (x_max - x_min) > 100 and (y_max - y_min) > 100):
                        
                        print(f"[DEBUG] Feature matching found region: ({x_min},{y_min}) to ({x_max},{y_max})")
                        return image[y_min:y_max, x_min:x_max]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Feature-based template matching failed: {e}")
            return None

    def _match_template_structure(self, image):
        """Structural similarity-based template matching"""
        try:
            try:
                from skimage.metrics import structural_similarity as ssim
            except ImportError:
                print("[DEBUG] scikit-image not available, skipping structural similarity")
                return None
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
            
            template_h, template_w = template_gray.shape
            best_score = 0
            best_region = None
            
            # Sliding window approach with different scales
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]
            
            for scale in scales:
                scaled_w = int(template_w * scale)
                scaled_h = int(template_h * scale)
                
                if scaled_w > image_gray.shape[1] or scaled_h > image_gray.shape[0]:
                    continue
                
                scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h))
                
                # Slide window across image
                step_size = max(20, min(scaled_w, scaled_h) // 10)
                
                for y in range(0, image_gray.shape[0] - scaled_h, step_size):
                    for x in range(0, image_gray.shape[1] - scaled_w, step_size):
                        window = image_gray[y:y+scaled_h, x:x+scaled_w]
                        
                        # Calculate SSIM
                        score = ssim(scaled_template, window)
                        
                        if score > best_score:
                            best_score = score
                            best_region = (x, y, scaled_w, scaled_h)
            
            if best_region is not None and best_score > 0.3:  # SSIM threshold
                x, y, w, h = best_region
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                print(f"[DEBUG] Structural matching found region: score={best_score:.3f}, pos=({x},{y}), size={w}x{h}")
                return image[y:y+h, x:x+w]
            
            return None
            
        except ImportError:
            print("[DEBUG] scikit-image not available, skipping structural similarity")
            return None
        except Exception as e:
            print(f"[DEBUG] Structural template matching failed: {e}")
            return None

    def _detect_by_ui_elements(self, image):
        """Original UI element detection method (fallback when no template)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for rectangular regions that could be the main interface
            # Use multiple edge detection methods
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            edges3 = cv2.Canny(gray, 80, 200)
            
            # Combine edge maps
            combined_edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
            
            # Morphological operations to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Too small
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it looks like a screen/interface
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 3.0:  # More flexible aspect ratio
                    # Calculate how much of the image this covers
                    coverage = area / (image.shape[0] * image.shape[1])
                    if 0.1 < coverage < 0.9:  # Reasonable coverage
                        candidates.append((contour, area, x, y, w, h, coverage))
            
            if candidates:
                # Sort by area and pick the best candidate
                candidates.sort(key=lambda x: x[1], reverse=True)
                _, _, x, y, w, h, _ = candidates[0]
                
                # Add some padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                return image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] UI element detection failed: {e}")
            return None

    def _detect_by_improved_color_analysis(self, image):
        """Improved color-based detection with adaptive thresholds"""
        try:
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Analyze image brightness to adapt thresholds
            mean_brightness = np.mean(gray)
            print(f"[DEBUG] Image mean brightness: {mean_brightness}")
            
            if mean_brightness > 120:  # Bright image
                lower_light = np.array([0, 0, 120])  # Reduced from 180 to 120
                upper_light = np.array([180, 80, 255])  # Increased saturation
            elif mean_brightness > 80:  # Medium brightness
                lower_light = np.array([0, 0, 80])   # Reduced from 120 to 80
                upper_light = np.array([180, 100, 255])  # Increased saturation
            else:  # Dark image
                lower_light = np.array([0, 0, 40])   # Reduced from 80 to 40
                upper_light = np.array([180, 120, 255])  # Increased saturation
            
            print(f"[DEBUG] Using HSV range: {lower_light} to {upper_light}")
            
            # Create mask for interface area
            light_mask = cv2.inRange(hsv, lower_light, upper_light)
            
            # Use LAB color space for better separation
            l_channel = lab[:,:,0]
            l_thresh = cv2.threshold(l_channel, 80, 255, cv2.THRESH_BINARY)[1]  # Reduced from 100 to 80
            
            # Combine masks
            combined_mask = cv2.bitwise_or(light_mask, l_thresh)
            
            cv2.imwrite("debug_mask.jpg", combined_mask)
            print(f"[DEBUG] Mask coverage: {np.sum(combined_mask > 0) / combined_mask.size * 100:.1f}%")
            
            kernel_size = max(5, min(15, int(min(image.shape[:2]) / 80)))  # Slightly larger kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Additional erosion to remove thin borders
            erosion_kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.erode(combined_mask, erosion_kernel, iterations=1)
            combined_mask = cv2.dilate(combined_mask, erosion_kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # More flexible area constraints
            img_area = image.shape[0] * image.shape[1]
            min_area = img_area * 0.03  # Reduced from 5% to 3%
            max_area = img_area * 0.98  # Increased from 95% to 98%
            
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < min_area or area > max_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # More flexible aspect ratio (DAT can be various shapes)
                if aspect_ratio < 0.5 or aspect_ratio > 4.0:
                    continue
                
                # edge_margin = 10  # Reduced from 30 to 10
                # if x < edge_margin or y < edge_margin or (x + w) > (image.shape[1] - edge_margin) or (y + h) > (image.shape[0] - edge_margin):
                #     continue
                
                # Improved scoring - focus on area and aspect ratio, not center position
                area_score = min(1.0, area / (img_area * 0.5))  # Normalize area score
                
                # Aspect ratio score (prefer landscape but not too strict)
                aspect_score = 1.0 if 1.0 <= aspect_ratio <= 2.0 else 0.7
                
                # Combined score - prioritize area and aspect ratio
                score = area_score * 0.7 + aspect_score * 0.3
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
            
            if best_contour is not None and best_score > 0.2:  # Reduced threshold from 0.3 to 0.2
                x, y, w, h = cv2.boundingRect(best_contour)
                
                padding = max(1, min(5, int(min(image.shape[:2]) / 200)))  # Reduced padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                print(f"[DEBUG] Color detection: ({x}, {y}) size {w}x{h}, score: {best_score:.3f}")
                return image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Color analysis failed: {e}")
            return None

    def _detect_by_text_regions(self, image):
        """Detect DAT interface by looking for characteristic text"""
        try:
            # Quick OCR scan to find DAT-related text
            result = self.ocr.predict(image)
            
            if not result or len(result) == 0:
                return None
            
            ocr_data = result[0]
            if not isinstance(ocr_data, dict) or 'rec_texts' not in ocr_data or 'rec_boxes' not in ocr_data:
                return None
            
            texts = ocr_data['rec_texts']
            boxes = ocr_data['rec_boxes']
            
            # Look for DAT-specific keywords
            dat_keywords = ['DAT', 'PHIÊN HỌC', 'km/h', 'TRUNG TÂM', 'HỌC VIÊN', 'GIÁO VIÊN']
            
            relevant_boxes = []
            for i, text in enumerate(texts):
                text_upper = text.upper()
                for keyword in dat_keywords:
                    if keyword in text_upper:
                        relevant_boxes.append(boxes[i])
                        break
            
            if len(relevant_boxes) >= 2:  # Need at least 2 DAT-related texts
                # Find bounding box of all relevant text regions
                all_points = np.concatenate(relevant_boxes)
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
        """Adaptive contour detection with multiple parameters"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try multiple threshold values
            threshold_values = [100, 127, 150, 180]
            best_result = None
            best_score = 0
            
            for thresh_val in threshold_values:
                # Binary threshold
                _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    img_area = image.shape[0] * image.shape[1]
                    
                    if area < img_area * 0.1 or area > img_area * 0.8:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 0.7 < aspect_ratio < 3.0:
                        # Score this detection
                        coverage = area / img_area
                        score = coverage * aspect_ratio if aspect_ratio > 1 else coverage / aspect_ratio
                        
                        if score > best_score:
                            best_score = score
                            best_result = (x, y, w, h)
            
            if best_result is not None:
                x, y, w, h = best_result
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                print(f"[DEBUG] Adaptive contours: ({x}, {y}) size {w}x{h}, score: {best_score:.3f}")
                return image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Adaptive contours failed: {e}")
            return None

    def _smart_fallback_crop(self, image):
        """Smart fallback that finds full screen content instead of center cropping"""
        try:
            h, w = image.shape[:2]
            
            # Method 1: Find the largest bright rectangular region
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to find bright areas
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            # Find contours of bright areas
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour that could be the screen
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Check if this looks like a reasonable screen area
                area_ratio = (w * h) / (image.shape[0] * image.shape[1])
                aspect_ratio = w / h
                
                if area_ratio > 0.1 and 0.5 < aspect_ratio < 4.0:
                    print(f"[DEBUG] Found bright region: ({x}, {y}) size {w}x{h}")
                    return image[y:y+h, x:x+w]
            
            # Method 2: Edge-based detection to find screen boundaries
            edges = cv2.Canny(gray, 50, 150)
            
            # Find horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines
            lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours from lines
            line_contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if line_contours:
                # Find bounding box of all line contours
                all_points = np.vstack([contour for contour in line_contours])
                x, y, w, h = cv2.boundingRect(all_points)
                
                # Add some padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                area_ratio = (w * h) / (image.shape[0] * image.shape[1])
                if area_ratio > 0.2:  # Reasonable screen area
                    print(f"[DEBUG] Edge-based detection: ({x}, {y}) size {w}x{h}")
                    return image[y:y+h, x:x+w]
            
            # Remove only obvious black borders from edges
            border_size = min(20, min(h, w) // 20)  # Max 20px or 5% of image
            
            print(f"[DEBUG] Full image with border removal: border_size={border_size}")
            return image[border_size:h-border_size, border_size:w-border_size]
                
        except Exception as e:
            print(f"[DEBUG] Smart fallback failed: {e}")
            return image

    def normalize_dat_image(self, dat_image):
        """Normalize DAT interface to standard size"""
        try:
            # Resize to standard size while maintaining aspect ratio
            h, w = dat_image.shape[:2]
            target_w, target_h = self.standard_size
            
            # Calculate scaling factor to fit within target size
            scale_w = target_w / w
            scale_h = target_h / h
            scale = min(scale_w, scale_h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(dat_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create a canvas of target size and center the resized image
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            canvas.fill(240)  # Light gray background
            
            # Calculate position to center the image
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            
            # Place resized image on canvas
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            print(f"[DEBUG] Normalized DAT image from {w}x{h} to {target_w}x{target_h}")
            return canvas
            
        except Exception as e:
            print(f"[DEBUG] Error in normalization: {e}")
            # Fallback: simple resize
            return cv2.resize(dat_image, self.standard_size, interpolation=cv2.INTER_AREA)

    def preprocess_normalized_image(self, image):
        """Preprocessing for normalized DAT interface image"""
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        processed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(processed, (1, 1), 0)
        
        return processed
    
    def extract_text_from_region(self, image, region_name, bbox):
        """Extract text from a specific region of the image"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Crop the region
            region_image = image[y1:y2, x1:x2]
            
            if region_image.size == 0:
                print(f"[DEBUG] Empty region for {region_name}")
                return ""
            
            # Apply additional preprocessing for small regions
            if region_image.shape[0] < 50 or region_image.shape[1] < 100:
                # Resize small regions for better OCR
                scale_factor = 2
                region_image = cv2.resize(region_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Run OCR on the region
            result = self.ocr.predict(region_image)
            
            print(f"[DEBUG] Region {region_name} OCR result: {result}")
            
            if not result or len(result) == 0:
                return ""
            
            ocr_data = result[0]
            if isinstance(ocr_data, dict) and 'rec_texts' in ocr_data:
                texts = ocr_data['rec_texts']
                if texts:
                    # Join all texts from this region
                    raw_text = " ".join(texts).strip()
                    # Clean the text to remove common OCR artifacts
                    return self.clean_ocr_text(raw_text, region_name)
            
            return ""
            
        except Exception as e:
            print(f"[DEBUG] Error processing region {region_name}: {e}")
            return ""
    
    def clean_ocr_text(self, text, region_name):
        """Clean OCR text to remove common artifacts and prefixes"""
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # Remove common prefixes that OCR might add
        prefixes_to_remove = [
            'hv:', 'mahv:', '4v:', 's24', 'loading...', 'hang het ha', 'hang hét c1', 
            'hang', 'hét', 'c1', 'loading', 'ma hv:', 'ma hv', 'hv ', 'mahv ', '4v ',
            'mahv:79335-20250513082447007', 'adng hien trang', 'khoá: k06a c1',
            'khoá:', 'k06a', 'c1', 'hang hét', 'hang het', 'khoa:', 'khoa ',
            'u i khoá:', 'u i khoa:', 'erzura', 'hang hét c1', 'hang het c1'
        ]
        
        # Remove prefixes (case insensitive)
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                text_lower = text.lower()
            # Also check if prefix appears anywhere in the text
            if prefix.lower() in text_lower:
                text = text.replace(prefix, '').strip()
                text_lower = text.lower()
        
        # Remove common OCR artifacts
        artifacts_to_remove = [
            'hang het ha', 'hang hét c1', 'hang hét', 'hang het',
            'adng hien trang', 'erzura', 'mu4', 'ên', 'uen', 'phiên 7'
        ]
        
        for artifact in artifacts_to_remove:
            if artifact.lower() in text_lower:
                text = text.replace(artifact, '').strip()
                text_lower = text.lower()
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def process_single_image(self, image_path):
        """Process a single image using region-based OCR"""
        print(f"Processing: {image_path}")
        
        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            dat_interface = self.detect_dat_interface(original_image)
            
            normalized_dat = self.normalize_dat_image(dat_interface)
            
            processed_image = self.preprocess_normalized_image(normalized_dat)
            
            # Extract information from each defined region
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

    def _remove_black_borders(self, image):
        """Remove black borders around the screen content with improved precision"""
        try:
            print(f"[DEBUG] Removing black borders from image size: {image.shape}")
            
            # Method 1: Precise content area detection using high brightness threshold
            result = self._detect_precise_content_area(image)
            if result is not None:
                print("[DEBUG] Successfully used precise content area detection")
                return result
            
            # Method 2: White/light content detection using LAB color space
            result = self._detect_by_white_content(image)
            if result is not None:
                print("[DEBUG] Successfully used white content detection")
                return result
            
            # Method 3: Improved statistical analysis
            result = self._detect_by_improved_statistics(image)
            if result is not None:
                print("[DEBUG] Successfully used improved statistical analysis")
                return result
            
            # Fallback: Original method with tighter constraints
            return self._original_border_removal_tight(image)
            
        except Exception as e:
            print(f"[DEBUG] Error in black border removal: {e}")
            return image

    def _detect_precise_content_area(self, image):
        """Detect content area using lower brightness threshold to include student photos"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # Reduced from 150 to 100
            
            print(f"[DEBUG] Binary threshold coverage: {np.sum(binary > 0) / binary.size * 100:.1f}%")
            cv2.imwrite("debug_binary.jpg", binary)
            
            # Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest bright contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > (h * w * 0.15):  # Reduced from 20% to 15%
                    x, y, cw, ch = cv2.boundingRect(largest_contour)
                    
                    padding = 2
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    cw = min(w - x, cw + 2 * padding)
                    ch = min(h - y, ch + 2 * padding)
                    
                    if cw > w * 0.3 and ch > h * 0.3:  # Reduced from 0.4 to 0.3
                        cropped = image[y:y+ch, x:x+cw]
                        print(f"[DEBUG] Precise content detection: cropped to ({x}, {y}) size {cw}x{ch}")
                        return cropped
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Precise content detection failed: {e}")
            return None

    def _detect_by_white_content(self, image):
        """Detect content area by focusing on white/light gray regions using LAB color space"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]  # Lightness channel
            h, w = l_channel.shape
            
            # Create mask for very light content (L > 180 in LAB space)
            light_mask = cv2.threshold(l_channel, 180, 255, cv2.THRESH_BINARY)[1]
            
            # Also check for light gray content (L > 150)
            gray_mask = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)[1]
            
            # Combine masks but prioritize very light content
            kernel = np.ones((3, 3), np.uint8)
            light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in light mask first
            contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            best_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > best_area and area > (h * w * 0.15):  # At least 15% of image
                    x, y, cw, ch = cv2.boundingRect(contour)
                    # Check if it's a reasonable rectangle
                    if cw > w * 0.3 and ch > h * 0.3:
                        best_contour = contour
                        best_area = area
            
            if best_contour is not None:
                x, y, cw, ch = cv2.boundingRect(best_contour)
                
                padding = 1
                x = max(0, x - padding)
                y = max(0, y - padding)
                cw = min(w - x, cw + 2 * padding)
                ch = min(h - y, ch + 2 * padding)
                
                cropped = image[y:y+ch, x:x+cw]
                print(f"[DEBUG] White content detection: cropped to ({x}, {y}) size {cw}x{ch}")
                return cropped
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] White content detection failed: {e}")
            return None

    def _detect_by_improved_statistics(self, image):
        """Improved statistical analysis with lower thresholds to include left side"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            row_means = np.mean(gray, axis=1)
            row_threshold = np.mean(row_means) * 0.4  # Reduced from 0.6 to 0.4
            
            print(f"[DEBUG] Row threshold: {row_threshold:.1f}, mean: {np.mean(row_means):.1f}")
            
            # Find content rows
            bright_rows = np.where(row_means > row_threshold)[0]
            if len(bright_rows) == 0:
                print("[DEBUG] No bright rows found")
                return None
            
            top_row = bright_rows[0]
            bottom_row = bright_rows[-1]
            
            # Column-wise analysis with same approach
            col_means = np.mean(gray, axis=0)
            col_threshold = np.mean(col_means) * 0.4  # Reduced from 0.6 to 0.4
            
            print(f"[DEBUG] Col threshold: {col_threshold:.1f}, mean: {np.mean(col_means):.1f}")
            
            bright_cols = np.where(col_means > col_threshold)[0]
            if len(bright_cols) == 0:
                print("[DEBUG] No bright columns found")
                return None
            
            left_col = bright_cols[0]
            right_col = bright_cols[-1]
            
            print(f"[DEBUG] Detected region: ({left_col}, {top_row}) to ({right_col}, {bottom_row})")
            
            # Validate the detected region
            new_w = right_col - left_col
            new_h = bottom_row - top_row
            
            if new_w > w * 0.3 and new_h > h * 0.3:  # Reduced from 0.4 to 0.3
                cropped = image[top_row:bottom_row, left_col:right_col]
                print(f"[DEBUG] Improved statistical analysis: cropped to ({left_col}, {top_row}) size {new_w}x{new_h}")
                return cropped
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Improved statistical analysis failed: {e}")
            return None

    def _original_border_removal_tight(self, image):
        """Original method with tighter constraints as final fallback"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > (h * w * 0.2):  # Increased from 0.1 to 0.2
                    x, y, cw, ch = cv2.boundingRect(largest_contour)
                    
                    padding = 1
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    cw = min(w - x, cw + 2 * padding)
                    ch = min(h - y, ch + 2 * padding)
                    
                    if cw > w * 0.4 and ch > h * 0.4:  # Increased from 0.3 to 0.4
                        cropped = image[y:y+ch, x:x+cw]
                        print(f"[DEBUG] Tight fallback: cropped to ({x}, {y}) size {cw}x{ch}")
                        return cropped
            
            print("[DEBUG] All border removal methods failed, returning original")
            return image
            
        except Exception as e:
            print(f"[DEBUG] Tight fallback failed: {e}")
            return image

def test_visualization():
    processor = SimpleOCRProcessor()
    
    # Test with dat_000.jpg
    test_image = "images/dat_013.jpg"  # Replace with your image path
    if os.path.exists(test_image):
        print("Running visualization test...")
        images = processor.visualize_processing_steps(test_image)
        
        if images:
            print("\nProcessing steps completed successfully!")
            print("Check the 'debug_images' folder for visualization results.")
        else:
            print("Visualization failed!")
    else:
        print(f"Test image {test_image} not found")

# Test function
def test_single_image():
    processor = SimpleOCRProcessor()
    
    # Test with dat_000.jpg
    test_image = "images/dat_016.jpg"  # Replace with your image path
    if os.path.exists(test_image):
        result = processor.process_single_image(test_image)
        print("\nExtracted Information:")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print(f"Test image {test_image} not found")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Run OCR extraction test")
    print("2. Run visualization test")
    test_visualization()

    # choice = input("Enter choice (1 or 2): ").strip()
    
    # if choice == "2":
    #     test_visualization()
    # else:
    #     test_single_image()
