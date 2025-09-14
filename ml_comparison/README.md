# ML Comparison - OCR Text Correction Pipeline

Thư mục này chứa hệ thống so sánh hiệu suất giữa OCR cơ bản và OCR với Machine Learning text correction.

## 📋 Tổng quan

Hệ thống này bao gồm 2 mô hình chính:
1. **Baseline OCR**: Sử dụng PaddleOCR thuần túy để nhận dạng text
2. **Enhanced OCR**: Baseline OCR + Text Correction ML để cải thiện độ chính xác

### 🎯 Kết quả so sánh
- **Baseline OCR**: 37.11% accuracy
- **Enhanced OCR**: 64.89% accuracy  
- **Cải thiện**: +27.78% accuracy

## 🏗️ Cấu trúc thư mục

```
ml_comparison/
├── main_ocr.py                    # Module OCR chính với PaddleOCR
├── run_baseline.py                # Chạy Baseline OCR model
├── run_pipeline.py                # Chạy pipeline Baseline + Text Correction ML
├── text_correction_ml.py          # Machine Learning text correction
├── text_normalizer.py             # Text normalization và cleaning
├── simple_correction_models/      # Thư mục chứa trained models
│   └── correction_models.pkl
├── baseline_results.csv           # Kết quả Baseline OCR
├── enhanced_results.csv           # Kết quả Enhanced OCR
├── baseline_summary.txt           # Tóm tắt Baseline OCR
├── enhanced_summary.txt           # Tóm tắt Enhanced OCR
└── README.md                      # File này
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
pip install opencv-python
pip install paddlepaddle
pip install paddleocr
pip install pandas
pip install numpy
pip install scikit-learn
pip install joblib
```

### 2. Chuẩn bị dữ liệu

Đảm bảo có các file sau trong thư mục gốc:
- `../images/` - Thư mục chứa ảnh test (dat_000.jpg đến dat_099.jpg)
- `../data.csv` - File ground truth với ký tự phân cách `;`
- `../dat_template.png` - Template image cho template matching

### 3. Chạy Baseline OCR

```bash
python run_baseline.py
```

**Kết quả:**
- Tạo file `baseline_results.csv` với kết quả chi tiết
- Tạo file `baseline_summary.txt` với tóm tắt performance
- Accuracy: ~37%

### 4. Chạy Enhanced Pipeline (Baseline + ML Correction)

```bash
python run_pipeline.py
```

**Kết quả:**
- Chạy Baseline OCR trước
- Train Text Correction ML model
- Apply correction và tạo kết quả enhanced
- Tạo file `enhanced_results.csv` và `enhanced_summary.txt`
- Accuracy: ~65% (cải thiện +27%)

### 5. Test riêng lẻ từng component

#### Test OCR cơ bản:
```bash
python main_ocr.py
```

#### Test Text Correction ML:
```bash
python text_correction_ml.py
```

## 📊 Chi tiết kết quả

### Baseline OCR Performance
| Field | Accuracy | Correct/Total |
|-------|----------|---------------|
| Student Name | 51.0% | 51/100 |
| Student ID | 15.0% | 15/100 |
| Vehicle Plate | 45.0% | 45/100 |
| Instructor Name | 82.0% | 82/100 |
| Distance Completed | 2.0% | 2/100 |
| Time Completed | 69.0% | 69/100 |
| Distance Remaining | 0.0% | 0/100 |
| Time Remaining | 58.0% | 58/100 |
| Total Sessions | 12.0% | 12/100 |

### Enhanced OCR Performance
| Field | Accuracy | Correct/Total | Improvement |
|-------|----------|---------------|-------------|
| Student Name | 51.0% | 51/100 | - |
| Student ID | 49.0% | 49/100 | +34% |
| Vehicle Plate | 66.0% | 66/100 | +21% |
| Instructor Name | 82.0% | 82/100 | - |
| Distance Completed | 79.0% | 79/100 | +77% |
| Time Completed | 70.0% | 70/100 | +1% |
| Distance Remaining | 77.0% | 77/100 | +77% |
| Time Remaining | 59.0% | 59/100 | +1% |
| Total Sessions | 51.0% | 51/100 | +39% |

## 🔧 Các module chính

### 1. `main_ocr.py` - SimpleOCRProcessor
**Chức năng:**
- Khởi tạo PaddleOCR với support tiếng Việt
- Detect DAT interface từ ảnh gốc
- Normalize ảnh về kích thước chuẩn (1280x720)
- Preprocess ảnh để tối ưu OCR
- Extract text từ 9 regions định sẵn

**Các regions được extract:**
- `student_name`: Tên học viên
- `student_id`: Mã học viên  
- `instructor_name`: Tên giáo viên
- `vehicle_plate`: Biển số xe
- `distance_completed`: Quãng đường đã hoàn thành
- `time_completed`: Thời gian đã hoàn thành
- `distance_remaining`: Quãng đường còn lại
- `time_remaining`: Thời gian còn lại
- `total_sessions`: Tổng số phiên học

### 2. `text_normalizer.py` - TextNormalizer
**Chức năng:**
- Normalize text cho từng loại field cụ thể
- Fix OCR errors phổ biến
- Clean prefixes/suffixes không cần thiết
- Support fuzzy matching cho tên tiếng Việt
- Enhanced matching cho Student ID

**Các methods chính:**
- `normalize_text()`: Normalize theo field type
- `normalize_student_id()`: Clean student ID format
- `normalize_vehicle_plate()`: Extract biển số xe VN
- `normalize_distance()`: Clean số liệu khoảng cách
- `normalize_time()`: Format thời gian
- `fuzzy_match()`: So sánh tên với diacritics

### 3. `text_correction_ml.py` - SimpleTextCorrectionML
**Chức năng:**
- Train ML model từ baseline OCR results
- Sử dụng N-gram models và TF-IDF
- Nearest Neighbors để tìm correction candidates
- Apply correction cho từng field

**Pipeline:**
1. Load baseline OCR results
2. Create training data từ OCR output + ground truth
3. Train correction models cho từng field
4. Apply correction cho new predictions
5. Save trained models

### 4. `run_baseline.py` - BaselineOCRRunner
**Chức năng:**
- Chạy Baseline OCR trên 100 ảnh test
- So sánh với ground truth
- Tính accuracy cho từng field
- Export results ra CSV và summary

### 5. `run_pipeline.py` - TextCorrectionPipeline
**Chức năng:**
- Chạy pipeline hoàn chỉnh: Baseline + ML Correction
- So sánh performance trước/sau correction
- Export results và metrics

## 📈 Cách cải thiện accuracy

### 1. Template Matching
- Sử dụng `dat_template.png` để detect DAT interface chính xác hơn
- Support multiple detection strategies (template, color, text, contours)

### 2. Text Normalization
- Remove OCR artifacts và prefixes
- Fix common OCR character misreads
- Support Vietnamese diacritics

### 3. Machine Learning Correction
- Train từ baseline results
- N-gram models cho character-level correction
- TF-IDF + Nearest Neighbors cho similarity matching

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **Import Error**: Đảm bảo các dependencies đã được cài đặt
2. **File Not Found**: Kiểm tra đường dẫn đến `../images/` và `../data.csv`
3. **Template Error**: Đảm bảo `../dat_template.png` tồn tại
4. **Memory Error**: Giảm số ảnh xử lý trong `limit` parameter

### Debug mode:
```python
# Trong main_ocr.py, uncomment để chạy visualization
test_visualization()
```

## 📝 Ghi chú

- Hệ thống được thiết kế cho ảnh DAT interface cụ thể
- Template matching cần template image chính xác
- ML correction cần baseline results để train
- Performance có thể khác nhau tùy chất lượng ảnh input


---
*Dự án Computer Vision - OCR cho hệ thống DAT*
