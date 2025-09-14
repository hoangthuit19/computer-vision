#!/usr/bin/env python3
"""
Text Normalization Module
Cung cấp các functions để normalize text cho từng field type
"""

import re

class TextNormalizer:
    """Class chứa tất cả các functions normalize text cho từng field"""
    
    def normalize_text(self, text, field_type=None):
        """Normalize text for comparison with field-specific cleaning"""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra spaces
        text = ' '.join(str(text).lower().split())
        
        # Remove common prefixes that OCR might add
        prefixes_to_remove = [
            'hv:', 'mahv:', '4v:', 's24', 'loading...', 'hang het ha', 'hang hét c1', 
            'hang', 'hét', 'c1', 'loading', 'ma hv:', 'ma hv', 'hv ', 'mahv ', '4v ',
            'mahv:79335-20250513082447007', 'adng hien trang', 'khoá: k06a c1',
            'khoá:', 'k06a', 'c1', 'hang hét', 'hang het'
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
            # Also check if prefix appears anywhere in the text
            if prefix.lower() in text:
                text = text.replace(prefix.lower(), '').strip()
        
        # Field-specific normalization
        if field_type == 'student_id':
            return self.normalize_student_id(text)
        elif field_type == 'vehicle_plate':
            return self.normalize_vehicle_plate(text)
        elif field_type in ['distance_completed', 'distance_remaining']:
            return self.normalize_distance(text)
        elif field_type in ['time_completed', 'time_remaining']:
            return self.normalize_time(text)
        elif field_type == 'total_sessions':
            return self.normalize_sessions(text)
        else:
            # General normalization for names and other fields
            return self.normalize_names(text)
    
    def normalize_student_id(self, text):
        """Normalize student ID - fix OCR errors in numbers with enhanced logic"""
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text = str(text).lower().strip()
        
        # Remove common prefixes that OCR might add
        prefixes_to_remove = [
            'hv:', 'mahv:', '4v:', 's24', 'loading...', 'hang het ha', 'hang hét c1', 
            'hang', 'hét', 'c1', 'loading', 'ma hv:', 'ma hv', 'hv ', 'mahv ', '4v ',
            'mahv:79335-20250513082447007', 'adng hien trang', 'khoá: k06a c1',
            'khoá:', 'k06a', 'c1', 'hang hét', 'hang het'
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix.lower()):
                text = text[len(prefix):].strip()
            # Also check if prefix appears anywhere in the text
            if prefix.lower() in text:
                text = text.replace(prefix.lower(), '').strip()
        
        # Remove all letters since student_id should only contain numbers and hyphens
        text = re.sub(r'[a-zA-Z]', '', text)
        
        # Replace colons with hyphens (common OCR error where : is used instead of -)
        text = text.replace(':', '-')
        
        # Remove all special characters except numbers and hyphens
        text = re.sub(r'[^\d\-]', '', text)
        
        # Clean up multiple consecutive hyphens
        text = re.sub(r'-+', '-', text)
        
        # Remove leading/trailing hyphens
        text = text.strip('-')
        
        # If the text looks like a name (contains letters), return empty
        if len(text) < 10:  # Student IDs should be long
            return ""
        
        return text.strip()
    
    def enhanced_student_id_match(self, pred_text, gt_text):
        """Enhanced matching for student IDs with fuzzy logic"""
        if not pred_text or not gt_text:
            return False
        
        # Normalize both texts
        pred_norm = self.normalize_student_id(pred_text)
        gt_norm = self.normalize_student_id(gt_text)
        
        # Exact match
        if pred_norm == gt_norm:
            return True
        
        # If either is empty after normalization, no match
        if not pred_norm or not gt_norm:
            return False
        
        # Check if one is contained in the other (for missing digits)
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return True
        
        # Check character-level similarity
        if len(pred_norm) == len(gt_norm):
            similar_chars = 0
            for c1, c2 in zip(pred_norm, gt_norm):
                if c1 == c2:
                    similar_chars += 1
            
            # If more than 85% characters match, consider it a match
            similarity = similar_chars / len(pred_norm) if len(pred_norm) > 0 else 0
            if similarity >= 0.95:
                return True
        
        return False
    
    def normalize_vehicle_plate(self, text):
        """Normalize vehicle plate - extract Vietnamese plate format 00X00000"""
        if not text:
            return ""
        
        # Convert to uppercase for processing
        text = str(text).upper().strip()
        
        # Remove common prefixes and suffixes that OCR might add
        prefixes_to_remove = [
            'HANG', 'HET', 'HA', 'GPTL', 'HANE', 'ERGOOG', 'B2', '2026', '123009'
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
            if text.endswith(prefix):
                text = text[:-len(prefix)].strip()
            # Also remove from anywhere in text
            text = text.replace(prefix, ' ').strip()
        
        # Remove extra spaces and split by spaces
        text = ' '.join(text.split())
        
        # First, try to find Vietnamese plate pattern 00X00000 in the text
        # Look for pattern: 2 digits + 1 letter + 5 digits
        plate_pattern = re.search(r'(\d{2}[A-Z]\d{5})', text)
        if plate_pattern:
            return plate_pattern.group(1)
        
        # If no exact pattern found, try to extract and clean the text
        # Remove all non-alphanumeric characters first
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        # Fix OCR errors for letters (common misreads)
        ocr_fixes = {
            'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '9', 'L': '1',
            'C': '0', 'D': '0', 'E': '6', 'A': '4', 'T': '7', 'N': '1',
            'M': '1', 'R': '1', 'U': '0', 'V': '1', 'W': '1', 'X': '1',
            'Y': '1', 'Z': '2', 'Q': '9'
        }
        
        # Apply OCR fixes
        for ocr_error, correct_char in ocr_fixes.items():
            text = text.replace(ocr_error, correct_char)
        
        # Now try to find the plate pattern again after OCR fixes
        plate_pattern = re.search(r'(\d{2}[A-Z]\d{5})', text)
        if plate_pattern:
            return plate_pattern.group(1)
        
        # If still no pattern found, return empty
        return ""
    
    def normalize_distance(self, text):
        """Normalize distance - extract numbers and decimal point with enhanced cleaning"""
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text = str(text).lower().strip()
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            'km', 'kr', 'k', 'm', 'distance', 'dist', 'd', 'le', 'hoc', 'sn la',
            'ahoc', 'ciso', 'gia', 'wd', 'ls', 'giao', 'vier', 'dem'
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
            if text.endswith(prefix):
                text = text[:-len(prefix)].strip()
            # Also remove from anywhere in text
            text = text.replace(prefix, ' ').strip()
        
        # Remove "km" specifically (common in OCR output)
        text = re.sub(r'\bkm\b', '', text).strip()
        
        # Remove time-like patterns (e.g., "00:00", ":59", "10:03")
        # But be careful not to remove decimal points
        text = re.sub(r':\d*', '', text)  # Remove colon and digits after it
        
        # Remove extra spaces and split by spaces
        text = ' '.join(text.split())
        
        # If multiple values, take the first one that looks like a number
        parts = text.split()
        for part in parts:
            # Check if part contains a number
            if re.search(r'\d', part):
                text = part
                break
        
        # Remove all letters since distance should only contain numbers and decimal point
        text = re.sub(r'[a-zA-Z]', '', text)
        
        # Extract numbers and decimal point only
        text = re.sub(r'[^\d\.]', '', text)
        
        # If empty or too short, return empty
        if not text or len(text) < 1:
            return ""
        
        # Validate that it's a reasonable distance (0-9999)
        try:
            distance = float(text)
            if distance < 0 or distance > 9999:
                return ""
        except ValueError:
            return ""
        
        return text.strip()
    
    def normalize_time(self, text):
        """Normalize time - extract numbers and colons with enhanced cleaning"""
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text = str(text).lower().strip()
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            'time', 't', 'h', 'm', 's', 'hour', 'minute', 'second', 'gio', 'phut', 'giay'
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
            if text.endswith(prefix):
                text = text[:-len(prefix)].strip()
            # Also remove from anywhere in text
            text = text.replace(prefix, ' ').strip()
        
        # Remove extra spaces and split by spaces
        text = ' '.join(text.split())
        
        # If multiple values, take the first one that looks like time
        parts = text.split()
        for part in parts:
            # Check if part contains time-like pattern (numbers and colons)
            if ':' in part or (len(part) >= 2 and any(c.isdigit() for c in part)):
                text = part
                break
        
        # Fix OCR errors for numbers
        ocr_fixes = {
            'o': '0', 'i': '1', 's': '5', 'b': '8', 'g': '9', 'l': '1',
            'c': '0', 'd': '0', 'e': '6', 'a': '4', 't': '7', 'n': '1',
            'm': '1', 'r': '1', 'u': '0', 'v': '1', 'w': '1', 'x': '1',
            'y': '1', 'z': '2', 'q': '9'
        }
        
        for ocr_error, correct_char in ocr_fixes.items():
            text = text.replace(ocr_error, correct_char)
        
        # Extract numbers and colons only
        text = re.sub(r'[^\d:]', '', text)
        
        # If empty or too short, return empty
        if not text or len(text) < 1:
            return ""
        
        # Validate time format (should have colons or be reasonable time)
        if ':' in text:
            # Time format like "12:34" or "12:34:56"
            parts = text.split(':')
            if len(parts) == 2:
                # Keep HH:MM format
                return text
            elif len(parts) == 3:
                # Convert HH:MM:SS to HH:MM
                return f"{parts[0]}:{parts[1]}"
        else:
            # Just numbers, validate reasonable time and format as HH:MM
            try:
                time_num = int(text)
                if 0 <= time_num <= 235959:  # Reasonable time range
                    # Format as HH:MM
                    if len(text) == 4:  # HHMM format
                        return f"{text[:2]}:{text[2:]}"
                    elif len(text) == 3:  # HMM format
                        return f"0{text[0]}:{text[1:]}"
                    elif len(text) == 2:  # MM format (assume minutes)
                        return f"00:{text}"
                    elif len(text) == 1:  # M format
                        return f"00:0{text}"
                    else:
                        return text
            except ValueError:
                pass
        
        return ""
    
    def normalize_sessions(self, text):
        """Normalize total sessions - extract numbers only using regex"""
        if not text:
            return ""
        
        # Convert to string and clean
        text = str(text).strip()
        
        # Use regex to extract ALL numbers from the text
        # This will find all sequences of digits and join them
        numbers = re.findall(r'\d+', text)
        
        if not numbers:
            return ""
        
        # Take the first number found (most likely to be the session count)
        session_number = numbers[0]
        
        # Remove leading zeros (e.g., "09" -> "9")
        session_number = session_number.lstrip('0') or '0'
        
        # Validate that it's a reasonable session count (1-20 for driving course)
        try:
            sessions = int(session_number)
            if sessions < 1 or sessions > 20:  # Driving course typically has 10-20 sessions
                return ""
        except ValueError:
            return ""
        
        return session_number
    
    def normalize_names(self, text):
        """Normalize names - fix Vietnamese characters and OCR errors"""
        # Fix common OCR character recognition errors for names
        # OCR often reads letters as numbers in names
        ocr_fixes = {
            '0': 'o',  # OCR reads o as 0
            '1': 'i',  # OCR reads i as 1  
            '5': 's',  # OCR reads s as 5
            '8': 'b',  # OCR reads b as 8
            '9': 'g',  # OCR reads g as 9
            '6': 'g',  # OCR reads g as 6
            '2': 'z',  # OCR reads z as 2
            '3': 'e',  # OCR reads e as 3
            '4': 'a',  # OCR reads a as 4
            '7': 't'   # OCR reads t as 7
        }
        
        for ocr_error, correct_char in ocr_fixes.items():
            text = text.replace(ocr_error, correct_char)
        
        # Normalize Vietnamese characters
        vietnamese_normalize = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd', 'Đ': 'd'
        }
        
        for old, new in vietnamese_normalize.items():
            text = text.replace(old, new)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def fuzzy_match(self, text1, text2):
        """Advanced fuzzy matching for Vietnamese text with diacritics"""
        if not text1 or not text2:
            return False
        
        # Convert to lowercase for comparison
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return True
        
        # Check if one is substring of another
        if text1 in text2 or text2 in text1:
            return True
        
        # Remove diacritics for comparison
        text1_no_diacritics = self.remove_diacritics(text1)
        text2_no_diacritics = self.remove_diacritics(text2)
        
        # Match without diacritics
        if text1_no_diacritics == text2_no_diacritics:
            return True
        
        # Check if one is substring of another (without diacritics)
        if text1_no_diacritics in text2_no_diacritics or text2_no_diacritics in text1_no_diacritics:
            return True
        
        # Calculate similarity with diacritics
        similarity_with_diacritics = self.calculate_similarity(text1, text2)
        if similarity_with_diacritics >= 0.9:
            return True
        
        # Calculate similarity without diacritics
        similarity_no_diacritics = self.calculate_similarity(text1_no_diacritics, text2_no_diacritics)
        if similarity_no_diacritics >= 0.9:
            return True
        
        return False
    
    def remove_diacritics(self, text):
        """Remove Vietnamese diacritics from text"""
        vietnamese_normalize = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd', 'Đ': 'd'
        }
        
        for viet_char, base_char in vietnamese_normalize.items():
            text = text.replace(viet_char, base_char)
        
        return text
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        common_chars = set(text1) & set(text2)
        total_chars = set(text1) | set(text2)
        
        if len(total_chars) == 0:
            return 0.0
        
        return len(common_chars) / len(total_chars)
    
    # def partial_name_match(self, text1, text2):
    #     """Partial name matching for Vietnamese names"""
    #     if not text1 or not text2:
    #         return False
        
    #     # Split into words and check if any word matches
    #     words1 = text1.split()
    #     words2 = text2.split()
        
    #     for word1 in words1:
    #         for word2 in words2:
    #             if len(word1) >= 3 and len(word2) >= 3:
    #                 if word1 in word2 or word2 in word1:
    #                     return True
        
    #     return False
    
    # def character_similarity_match(self, text1, text2):
    #     """Character-level similarity matching"""
    #     if not text1 or not text2:
    #         return False
        
    #     # Calculate character-level similarity
    #     max_len = max(len(text1), len(text2))
    #     if max_len == 0:
    #         return False
        
    #     similar_chars = 0
    #     for i in range(min(len(text1), len(text2))):
    #         if text1[i] == text2[i]:
    #             similar_chars += 1
        
    #     similarity = similar_chars / max_len
    #     return similarity >= 0.7

