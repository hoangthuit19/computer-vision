#!/usr/bin/env python3
"""
Simple Text Correction ML using Statistical Methods
Alternative approach using N-gram models and edit distance for OCR correction
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from text_normalizer import TextNormalizer
import warnings
warnings.filterwarnings('ignore')

class SimpleTextCorrectionML:
    def __init__(self):
        """Initialize Simple Text Correction ML"""
        self.normalizer = TextNormalizer()
        self.correction_models = {}
        self.ground_truth_patterns = defaultdict(list)
        self.ngram_models = {}
        
        print("🤖 Simple Text Correction ML initialized")
    
    def create_training_data(self, baseline_results_file=None):
        """Create training data from Baseline OCR results"""
        print("📊 Creating training data from Baseline OCR results...")
        
        # Use fixed baseline results file
        if baseline_results_file is None:
            baseline_results_file = 'baseline_results.csv'
            if not os.path.exists(baseline_results_file):
                print("❌ No baseline OCR results found")
                return None
            print(f"📁 Using baseline results: {baseline_results_file}")
        
        try:
            # Load baseline OCR results
            baseline_df = pd.read_csv(baseline_results_file)
            print(f"📊 Loaded {len(baseline_df)} baseline OCR results")
            
            # Load ground truth
            gt_df = pd.read_csv('../data.csv', sep=';')
            gt_df.columns = gt_df.columns.str.strip()
            
            # Create mapping
            gt_mapping = {}
            for _, row in gt_df.iterrows():
                image_name = row['Image Name']
                gt_mapping[image_name] = {
                    'student_name': str(row.get('Student Name', '')).strip(),
                    'student_id': str(row.get('Student ID', '')).strip(),
                    'instructor_name': str(row.get('Instructor Name', '')).strip(),
                    'vehicle_plate': str(row.get('Vehicle Plate', '')).strip(),
                    'distance_completed': str(row.get('Distance Completed (km)', '')).strip(),
                    'time_completed': str(row.get('Time Completed', '')).strip(),
                    'distance_remaining': str(row.get('Distance Remaining (km)', '')).strip(),
                    'time_remaining': str(row.get('Time Remaining', '')).strip(),
                    'total_sessions': str(row.get('Total Sessions', '')).strip()
                }
            
            # Prepare training data
            training_data = defaultdict(list)
            
            fields = ['student_name', 'student_id', 'instructor_name', 'vehicle_plate', 
                     'distance_completed', 'time_completed', 'distance_remaining', 
                     'time_remaining', 'total_sessions']
            
            for _, row in baseline_df.iterrows():
                image_name = row['image_name']
                
                if image_name not in gt_mapping:
                    continue
                
                gt_data = gt_mapping[image_name]
                
                for field in fields:
                    if f'predicted_{field}' in row and field in gt_data:
                        # Get OCR output (input) and ground truth (output)
                        ocr_text = str(row[f'predicted_{field}']).strip()
                        gt_text = gt_data[field]
                        
                        # Filter valid samples
                        if (ocr_text and gt_text and 
                            ocr_text != 'nan' and gt_text != 'nan' and
                            len(ocr_text) > 0 and len(gt_text) > 0):
                            
                            # Normalize texts
                            ocr_normalized = self.normalizer.normalize_text(ocr_text, field_type=field)
                            gt_normalized = self.normalizer.normalize_text(gt_text, field_type=field)
                            
                            # Store training data
                            training_data[field].append({
                                'input': ocr_normalized,
                                'output': gt_normalized
                            })
            
            print(f"✅ Created training data:")
            for field, data in training_data.items():
                print(f"   {field}: {len(data)} samples")
            
            return training_data
            
        except Exception as e:
            print(f"❌ Error creating training data: {e}")
            return None
    
    def build_ngram_model(self, texts, n=3):
        """Build N-gram model from texts"""
        ngrams = defaultdict(int)
        
        for text in texts:
            if isinstance(text, str):
                # Add padding
                padded_text = '#' * (n-1) + text + '#' * (n-1)
                
                # Extract n-grams
                for i in range(len(padded_text) - n + 1):
                    ngram = padded_text[i:i+n]
                    ngrams[ngram] += 1
        
        return ngrams
    
    def train_field_model(self, field, training_data):
        """Train correction model for a specific field"""
        print(f"🏋️ Training model for {field}...")
        
        if not training_data:
            print(f"   No training data for {field}")
            return
        
        # Extract inputs and outputs
        inputs = [item['input'] for item in training_data]
        outputs = [item['output'] for item in training_data]
        
        # Build N-gram models
        input_ngrams = self.build_ngram_model(inputs, n=3)
        output_ngrams = self.build_ngram_model(outputs, n=3)
        
        # Create correction mapping
        correction_map = {}
        for i, (input_text, output_text) in enumerate(zip(inputs, outputs)):
            if input_text != output_text:
                correction_map[input_text] = output_text
        
        # Build TF-IDF model for similarity matching
        all_texts = inputs + outputs
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=1000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Build nearest neighbors model
            nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
            nn_model.fit(tfidf_matrix)
            
            # Store model
            self.correction_models[field] = {
                'correction_map': correction_map,
                'vectorizer': vectorizer,
                'nn_model': nn_model,
                'all_texts': all_texts,
                'inputs': inputs,
                'outputs': outputs
            }
            
            print(f"   ✅ Model trained for {field}: {len(correction_map)} corrections")
            
        except Exception as e:
            print(f"   ❌ Error training {field}: {e}")
    
    def train(self, training_data):
        """Train all field models"""
        print("🚀 Training Simple Text Correction ML...")
        
        fields = ['student_name', 'student_id', 'instructor_name', 'vehicle_plate', 
                 'distance_completed', 'time_completed', 'distance_remaining', 
                 'time_remaining', 'total_sessions']
        
        for field in fields:
            if field in training_data:
                self.train_field_model(field, training_data[field])
        
        print("✅ Training completed!")
    
    def correct_text(self, text, field_type=None):
        """Correct text using trained models and text normalization"""
        if not text:
            return text
        
        # First apply text normalization for specific fields
        if field_type == 'total_sessions':
            # Use text normalizer to clean sessions field
            from text_normalizer import TextNormalizer
            normalizer = TextNormalizer()
            normalized_text = normalizer.normalize_sessions(text)
            if normalized_text:
                return normalized_text
        elif field_type in ['time_completed', 'time_remaining']:
            # Use text normalizer to clean time fields
            from text_normalizer import TextNormalizer
            normalizer = TextNormalizer()
            normalized_text = normalizer.normalize_time(text)
            if normalized_text:
                return normalized_text
        
        # Then apply ML correction if model exists
        if field_type not in self.correction_models:
            return text
        
        try:
            model = self.correction_models[field_type]
            
            # Direct mapping
            if text in model['correction_map']:
                return model['correction_map'][text]
            
            # Similarity-based correction
            text_vector = model['vectorizer'].transform([text])
            distances, indices = model['nn_model'].kneighbors(text_vector)
            
            # Find best match
            best_match = None
            best_score = 0
            
            for i, idx in enumerate(indices[0]):
                if idx < len(model['inputs']):
                    candidate_input = model['inputs'][idx]
                    candidate_output = model['outputs'][idx]
                    
                    # Calculate similarity
                    similarity = SequenceMatcher(None, text, candidate_input).ratio()
                    
                    if similarity > best_score and similarity > 0.6:
                        best_score = similarity
                        best_match = candidate_output
            
            if best_match:
                return best_match
            
            return text
            
        except Exception as e:
            print(f"❌ Error correcting text: {e}")
            return text
    
    def correct_baseline_results(self, baseline_results):
        """Apply text correction to baseline results"""
        print("🔧 Applying text correction to baseline results...")
        
        corrected_results = []
        
        # Handle tuple format (results, accuracy)
        if isinstance(baseline_results, tuple) and len(baseline_results) == 2:
            results_list = baseline_results[0]
            accuracy = baseline_results[1]
            print(f"📊 Processing {len(results_list)} results with accuracy: {accuracy}")
        else:
            results_list = baseline_results
        
        for result in results_list:
            if not isinstance(result, dict):
                print(f"⚠️ Skipping non-dict result: {type(result)}")
                continue
            corrected_result = result.copy()
            
            # Apply correction to each field
            fields = ['student_name', 'student_id', 'instructor_name', 'vehicle_plate', 
                     'distance_completed', 'time_completed', 'distance_remaining', 
                     'time_remaining', 'total_sessions']
            
            for field in fields:
                if field in result:
                    original_text = result[field]
                    corrected_text = self.correct_text(original_text, field_type=field)
                    corrected_result[field] = corrected_text
            
            corrected_results.append(corrected_result)
        
        print(f"✅ Corrected {len(corrected_results)} results")
        return corrected_results
    
    def save_model(self, model_dir='simple_correction_models'):
        """Save trained model"""
        import joblib
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save correction models
        joblib.dump(self.correction_models, os.path.join(model_dir, 'correction_models.pkl'))
        
        print(f"✅ Model saved to {model_dir}")
    
    def load_model(self, model_dir='simple_correction_models'):
        """Load trained model"""
        import joblib
        
        try:
            self.correction_models = joblib.load(os.path.join(model_dir, 'correction_models.pkl'))
            print(f"✅ Model loaded from {model_dir}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Test the Simple Text Correction ML"""
    print("🧪 Testing Simple Text Correction ML...")
    
    # Initialize model
    correction_ml = SimpleTextCorrectionML()
    
    # Create training data
    training_data = correction_ml.create_training_data()
    
    if training_data:
        # Train model
        correction_ml.train(training_data)
        
        # Test correction
        test_cases = [
            ("51d19675 hang het ha", "vehicle_plate"),
            ("nguyen anh minh", "student_name"),
            ("04.3 km", "distance_completed"),
            ("00:9", "time_completed"),
            ("bui anh tuan", "instructor_name")
        ]
        
        print("\n🔍 Testing text correction:")
        for text, field in test_cases:
            corrected = correction_ml.correct_text(text, field)
            print(f"   '{text}' ({field}) → '{corrected}'")
        
        # Save model
        correction_ml.save_model()
        
        print("\n🎉 Simple Text Correction ML training completed!")
    else:
        print("❌ No training data available")

if __name__ == "__main__":
    main()

