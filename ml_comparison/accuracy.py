import re
import pandas as pd

# ==============================
# Text Normalizer
# ==============================
class TextNormalizer:
    def __init__(self):
        self.prefixes = [
            "hv:", "mahv:", "4v:", "s24", "loading...", "hang het ha", "nhet hang",
            "ky", "hung", "ha", "trieu", "oppt", "chot", "ruot", "chang", "phu", "hop",
            "hpt", "hvhv", "hvh"
        ]

        self.ocr_fixes = str.maketrans({
            'O':'0', 'o':'0', 'I':'1', 'l':'1', 'S':'5', 's':'5',
            'B':'8', 'G':'9', 'Z':'2', 'Q':'9', 'D':'0', 'C':'0',
            'E':'6', 'A':'4', 'T':'7', 'N':'1', 'M':'1', 'R':'1',
            'U':'0', 'V':'1', 'W':'1', 'X':'1', 'Y':'1'
        })

    def remove_prefixes(self, text):
        if not text:
            return ""
        text = re.sub(r'^(?:' + "|".join(map(re.escape, self.prefixes)) + ")", "", text, flags=re.I)
        text = re.sub(r'(?:' + "|".join(map(re.escape, self.prefixes)) + r')$', "", text, flags=re.I)
        for p in self.prefixes:
            text = text.replace(p, "")
        return text.strip()

    def normalize_student_id(self, text):
        text = self.remove_prefixes(text).translate(self.ocr_fixes)
        text = re.sub(r'[^0-9\-]', '', text)
        text = re.sub(r'-+', '-', text).strip('-')
        return text

    def normalize_student_name(self, text):
        return self.remove_prefixes(text).strip().lower() if text else ""

    def normalize_vehicle_plate(self, text):
        text = self.remove_prefixes(text).upper().translate(self.ocr_fixes)
        match = re.search(r'\d{2}[A-Z]\d{5}', text)
        return match.group(0) if match else text

    def normalize_distance(self, text):
        text = self.remove_prefixes(text).translate(self.ocr_fixes)
        match = re.search(r'\d+(?:\.\d+)?', text)
        return match.group(0) if match else text

    def normalize_time(self, text):
        text = self.remove_prefixes(text).translate(self.ocr_fixes)
        match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', text)
        return match.group(1) if match else text

    def normalize_text(self, text, field):
        if not text:
            return ""
        if field == "student_id":
            return self.normalize_student_id(text)
        elif field in ["student_name", "instructor_name"]:
            return self.normalize_student_name(text)
        elif field == "vehicle_plate":
            return self.normalize_vehicle_plate(text)
        elif field in ["distance_completed", "distance_remaining"]:
            return self.normalize_distance(text)
        elif field in ["time_completed", "time_remaining"]:
            return self.normalize_time(text)
        return self.remove_prefixes(text).translate(self.ocr_fixes).strip()


# ==============================
# Accuracy Evaluator
# ==============================
class AccuracyEvaluator:
    def __init__(self, csv_path):
        self.gt_df = pd.read_csv(csv_path, sep=';', encoding="utf-8-sig")
        self.gt_df.columns = self.gt_df.columns.str.strip()
        self.normalizer = TextNormalizer()

    def calculate_accuracy(self, outputs, image_name):
        gt_row = self.gt_df[self.gt_df["Image Name"] == image_name]
        if len(gt_row) == 0:
            print(f"[WARN] Không tìm thấy ground truth cho {image_name}")
            return 0.0
        gt_row = gt_row.iloc[0]

        fields_to_check = {
            "student_name": "Student Name",
            "student_id": "Student ID",
            "vehicle_plate": "Vehicle Plate",
            "instructor_name": "Instructor Name",
            "distance_completed": "Distance Completed (km)",
            "time_completed": "Time Completed",
            "distance_remaining": "Distance Remaining (km)",
            "time_remaining": "Time Remaining",
            "total_sessions": "Total Sessions",
        }

        correct, total = 0, 0

        for pred in outputs:
            pred_field = pred["class"]
            pred_value = str(pred["ocr_text"]).strip()

            if pred_field in fields_to_check:
                gt_value = str(gt_row[fields_to_check[pred_field]]).strip()
                if gt_value and gt_value != "nan":
                    total += 1
                    # Normalize cả 2 bên
                    pred_norm = self.normalizer.normalize_text(pred_value, pred_field)
                    gt_norm = self.normalizer.normalize_text(gt_value, pred_field)

                    if pred_norm == gt_norm:
                        print(f"[OK] {pred_field}: pred='{pred_norm}' | gt='{gt_norm}'")
                        correct += 1
                    else:
                        print(f"[MISS] {pred_field}: pred='{pred_norm}' | gt='{gt_norm}'")

        acc = (correct / total * 100) if total > 0 else 0.0
        return acc
