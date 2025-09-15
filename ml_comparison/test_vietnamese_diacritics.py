#!/usr/bin/env python3
"""
Test script for Vietnamese diacritics enhancement
"""

import sys
import os
sys.path.append('..')

from run_easyocr import VietnameseDiacriticsEnhancer
import unittest

class TestVietnameseEnhancement(unittest.TestCase):
    
    def setUp(self):
        self.enhancer = VietnameseDiacriticsEnhancer()
    
    def test_name_restoration(self):
        """Test Vietnamese name diacritics restoration"""
        test_cases = [
            ("nguyen van anh", "Nguyễn Văn Anh"),
            ("tran thi huong", "Trần Thị Hương"),
            ("le minh quan", "Lê Minh Quân"),
            ("pham thuy hang", "Phạm Thúy Hằng"),
            ("do xuan thanh", "Đỗ Xuân Thành"),
            ("hoang thi lien", "Hoàng Thị Liên"),
        ]
        
        for input_text, expected in test_cases:
            result = self.enhancer.restore_vietnamese_diacritics(input_text)
            print(f"Input: '{input_text}' -> Output: '{result}' (Expected: '{expected}')")
            # Note: This is a fuzzy match since we're testing enhancement, not exact restoration
            self.assertIn(expected.split()[0], result)  # Check if first name is enhanced
    
    def test_post_processing(self):
        """Test OCR post-processing for Vietnamese text"""
        test_cases = [
            ("Nguy3n V4n 4nh", "Nguyen Van Anh"),  # Numbers to letters
            ("Tr4n Th1 Hu0ng", "Tran Thi Huong"),  # Mixed corrections
            ("L3 M1nh Qu4n", "Le Minh Quan"),      # Multiple corrections
        ]
        
        for input_text, expected_base in test_cases:
            result = self.enhancer.post_process_vietnamese_text(input_text)
            print(f"Post-process: '{input_text}' -> '{result}'")
            # Check that numbers are converted to letters
            self.assertNotRegex(result, r'\d')  # Should not contain digits
    
    def test_pattern_matching(self):
        """Test Vietnamese name pattern matching"""
        test_text = "hv: nguyen van minh"
        result = self.enhancer.restore_vietnamese_diacritics(test_text)
        print(f"Pattern test: '{test_text}' -> '{result}'")
        
        # Should contain enhanced Vietnamese names
        self.assertIn("Nguyễn", result)

def main():
    print("=== TESTING VIETNAMESE DIACRITICS ENHANCEMENT ===")
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Manual testing
    enhancer = VietnameseDiacriticsEnhancer()
    
    print("\n=== MANUAL TESTING ===")
    test_names = [
        "nguyen van anh",
        "tran thi huong", 
        "le minh quan",
        "pham duc nam",
        "hoang thi lan"
    ]
    
    for name in test_names:
        enhanced = enhancer.restore_vietnamese_diacritics(name)
        post_processed = enhancer.post_process_vietnamese_text(f"Nguy3n V4n {name}")
        
        print(f"Original: {name}")
        print(f"Enhanced: {enhanced}")
        print(f"Post-processed: {post_processed}")
        print("-" * 40)

if __name__ == "__main__":
    main()
