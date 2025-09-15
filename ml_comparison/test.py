import os
import sys
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO


# Add parent directory to path to import our modules
sys.path.append('..')

try:
    import easyocr
    from run_easyocr import VietnameseDiacriticsEnhancer
except ImportError as e:
    print(f"Import error: {e}")
    print("Installing required packages...")
    os.system("pip install easyocr opencv-python pillow")
    import easyocr
    from run_easyocr import VietnameseDiacriticsEnhancer

def download_image_from_url(url, save_path):
    """Download image from URL and save locally"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Save image
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Image downloaded successfully: {save_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return False

def extract_all_text_from_image(image_path):
    """Extract all text from the provided image using enhanced Vietnamese OCR"""
    
    print("🔍 TRÍCH XUẤT TOÀN BỘ KÝ TỰ TỪ HÌNH ẢNH")
    print("=" * 60)
    
    # Initialize EasyOCR with Vietnamese and English support
    print("📚 Khởi tạo EasyOCR với hỗ trợ tiếng Việt...")
    # reader = easyocr.Reader(['vi', 'en'], gpu=False)
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    
    # Initialize Vietnamese enhancer
    vietnamese_enhancer = VietnameseDiacriticsEnhancer()
    
    # Enhance image for better OCR
    print("🖼️  Cải thiện chất lượng hình ảnh...")
    enhanced_image_path = vietnamese_enhancer.enhance_image_for_ocr(image_path)
    
    # Perform OCR
    print("🔤 Đang thực hiện OCR...")
    ocr_results = reader.readtext(enhanced_image_path)
    
    print(f"📊 Tìm thấy {len(ocr_results)} vùng text")
    print("\n" + "=" * 60)
    print("📝 TOÀN BỘ KÝ TỰ ĐƯỢC TRÍCH XUẤT:")
    print("=" * 60)
    
    all_text_lines = []
    enhanced_text_lines = []
    
    # Sort results by vertical position (top to bottom)
    sorted_results = sorted(ocr_results, key=lambda x: min(point[1] for point in x[0]))
    
    for i, (box, text, confidence) in enumerate(sorted_results):
        # Get bounding box coordinates
        top_left = box[0]
        bottom_right = box[2]
        
        print(f"\n[{i+1:2d}] Vị trí: ({int(top_left[0])}, {int(top_left[1])}) -> ({int(bottom_right[0])}, {int(bottom_right[1])})")
        print(f"     Text gốc: '{text}'")
        print(f"     Độ tin cậy: {confidence:.3f}")
        
        # Apply Vietnamese enhancement
        enhanced_text = vietnamese_enhancer.post_process_vietnamese_text(text)
        if enhanced_text != text:
            print(f"     Text cải tiến: '{enhanced_text}' ✨")
            enhanced_text_lines.append(enhanced_text)
        else:
            enhanced_text_lines.append(text)
        
        all_text_lines.append(text)
    
    # Group text by lines (similar Y coordinates)
    def group_text_by_lines(results, threshold=30):
        """Group OCR results into lines based on Y coordinates"""
        if not results:
            return []
        
        # Create list of (avg_y, avg_x, text, enhanced_text)
        text_items = []
        for i, (box, text, confidence) in enumerate(results):
            avg_y = sum(point[1] for point in box) / 4
            avg_x = sum(point[0] for point in box) / 4
            enhanced_text = vietnamese_enhancer.post_process_vietnamese_text(text)
            text_items.append((avg_y, avg_x, text, enhanced_text))
        
        # Sort by Y coordinate
        text_items.sort(key=lambda x: x[0])
        
        # Group into lines
        lines = []
        current_line = []
        current_y = text_items[0][0] if text_items else 0
        
        for item in text_items:
            y, x, text, enhanced_text = item
            if abs(y - current_y) > threshold:
                if current_line:
                    # Sort current line by X coordinate and join
                    current_line.sort(key=lambda x: x[1])
                    line_text = ' '.join(item[2] for item in current_line)
                    enhanced_line_text = ' '.join(item[3] for item in current_line)
                    lines.append((line_text, enhanced_line_text))
                current_line = []
                current_y = y
            current_line.append(item)
        
        # Add last line
        if current_line:
            current_line.sort(key=lambda x: x[1])
            line_text = ' '.join(item[2] for item in current_line)
            enhanced_line_text = ' '.join(item[3] for item in current_line)
            lines.append((line_text, enhanced_line_text))
        
        return lines
    
    # Group text into lines
    text_lines = group_text_by_lines(ocr_results)
    
    print("\n" + "=" * 60)
    print("📄 TEXT ĐƯỢC NHÓM THEO DÒNG:")
    print("=" * 60)
    
    for i, (original_line, enhanced_line) in enumerate(text_lines):
        print(f"\nDòng {i+1}:")
        print(f"  Gốc: '{original_line}'")
        if enhanced_line != original_line:
            print(f"  Cải tiến: '{enhanced_line}' ✨")
    
    # Extract all unique text
    print("\n" + "=" * 60)
    print("📋 TẤT CẢ KÝ TỰ ĐƯỢC TRÍCH XUẤT (KHÔNG TRÙNG LẶP):")
    print("=" * 60)
    
    all_unique_text = set()
    all_enhanced_text = set()
    
    for original_line, enhanced_line in text_lines:
        all_unique_text.add(original_line.strip())
        all_enhanced_text.add(enhanced_line.strip())
    
    print("\n🔤 Text gốc từ OCR:")
    for i, text in enumerate(sorted(all_unique_text), 1):
        if text:  # Only show non-empty text
            print(f"  {i}. {text}")
    
    print("\n✨ Text sau khi cải tiến tiếng Việt:")
    for i, text in enumerate(sorted(all_enhanced_text), 1):
        if text:  # Only show non-empty text
            print(f"  {i}. {text}")
    
    # Analyze specific patterns
    print("\n" + "=" * 60)
    print("🔍 PHÂN TÍCH CÁC PATTERN ĐẶC BIỆT:")
    print("=" * 60)
    
    combined_text = ' '.join(enhanced_text_lines).lower()
    
    # Look for time patterns
    import re
    time_patterns = re.findall(r'\d{1,2}:\d{2}', combined_text)
    if time_patterns:
        print(f"⏰ Thời gian: {time_patterns}")
    
    # Look for date patterns
    date_patterns = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', combined_text)
    if date_patterns:
        print(f"📅 Ngày tháng: {date_patterns}")
    
    # Look for numbers
    number_patterns = re.findall(r'\d+\.?\d*', combined_text)
    if number_patterns:
        print(f"🔢 Các số: {number_patterns}")
    
    # Look for Vietnamese words
    vietnamese_words = []
    for text in enhanced_text_lines:
        words = text.split()
        for word in words:
            if re.search(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', word.lower()):
                vietnamese_words.append(word)
    
    if vietnamese_words:
        print(f"🇻🇳 Từ tiếng Việt có dấu: {list(set(vietnamese_words))}")
    
    # Clean up enhanced image
    if enhanced_image_path != image_path and os.path.exists(enhanced_image_path):
        try:
            os.remove(enhanced_image_path)
        except:
            pass
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH TRÍCH XUẤT TEXT!")
    print("=" * 60)
    
    return {
        'original_text_lines': [line[0] for line in text_lines],
        'enhanced_text_lines': [line[1] for line in text_lines],
        'all_text': list(all_unique_text),
        'enhanced_text': list(all_enhanced_text),
        'time_patterns': time_patterns,
        'date_patterns': date_patterns,
        'number_patterns': number_patterns,
        'vietnamese_words': list(set(vietnamese_words))
    }

def main():
    """Main function to process the provided image"""
    
    # Image URL from the user
    image_url = "/Users/thinhvp/Desktop/computer_vision/computer-vision/test_img/"
    local_image_path = "/Users/thinhvp/Desktop/computer_vision/computer-vision/test_img/test.jpg"
    
    print("🖼️  Đang tải hình ảnh từ URL...")
    
    results = extract_all_text_from_image(local_image_path)
        
        # Save results to file
    with open("extracted_text_results.txt", "w", encoding="utf-8") as f:
            f.write("TOÀN BỘ KÝ TỰ ĐƯỢC TRÍCH XUẤT TỪ HÌNH ẢNH\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("TEXT GỐC:\n")
            for i, text in enumerate(results['all_text'], 1):
                if text.strip():
                    f.write(f"{i}. {text}\n")
            
            f.write("\nTEXT SAU KHI CẢI TIẾN:\n")
            for i, text in enumerate(results['enhanced_text'], 1):
                if text.strip():
                    f.write(f"{i}. {text}\n")
            
            f.write(f"\nTHỜI GIAN: {results['time_patterns']}\n")
            f.write(f"NGÀY THÁNG: {results['date_patterns']}\n")
            f.write(f"CÁC SỐ: {results['number_patterns']}\n")
            f.write(f"TỪ TIẾNG VIỆT: {results['vietnamese_words']}\n")
        
    print(f"\n💾 Kết quả đã được lưu vào file: extracted_text_results.txt")
        
        # Clean up
    if os.path.exists(local_image_path):
        os.remove(local_image_path)
    
    # Download image
    # if download_image_from_url(image_url, local_image_path):
    #     # Extract text from image
        
    
    # else:
    #     print("❌ Không thể tải hình ảnh. Vui lòng kiểm tra URL.")

if __name__ == "__main__":
    main()
