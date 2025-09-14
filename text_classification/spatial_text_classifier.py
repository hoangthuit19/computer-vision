#!/usr/bin/env python3
"""
Spatial Text Classification for OCR Fields
Classify text snippets based on text content + spatial position + context
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import re
import os
import cv2

class SpatialTextClassifier:
    def __init__(self):
        self.text_model = None
        self.spatial_model = None
        self.combined_model = None
        self.tokenizer = None
        self.label_encoder = None
        self.scaler = None
        self.max_length = 50
        self.vocab_size = 1000
        
    def load_data(self):
        """Load training data with spatial information"""
        print("Loading spatial training data...")
        
        # Load baseline results
        baseline_df = pd.read_csv('../ml_comparison/baseline_results.csv')
        print(f"Loaded {len(baseline_df)} baseline results")
        
        # Create training data with spatial features
        training_data = []
        
        fields = ['student_name', 'student_id', 'vehicle_plate', 'instructor_name', 
                 'distance_completed', 'time_completed', 'distance_remaining', 
                 'time_remaining', 'total_sessions']
        
        for idx, row in baseline_df.iterrows():
            image_name = row['image_name']
            
            # Create spatial context for this image
            image_texts = []
            for field in fields:
                predicted_col = f'predicted_{field}'
                if predicted_col in row:
                    text = str(row[predicted_col]).strip()
                    if text != 'nan' and text:
                        # Simulate bounding box (in real case, get from OCR)
                        bbox = self.simulate_bbox(field, len(image_texts))
                        image_texts.append({
                            'text': text,
                            'field': field,
                            'bbox': bbox
                        })
            
            # Add spatial context
            for i, item in enumerate(image_texts):
                spatial_features = self.extract_spatial_features(item, image_texts, i)
                training_data.append({
                    'text': item['text'],
                    'field': item['field'],
                    'bbox': item['bbox'],
                    'spatial_features': spatial_features,
                    'image_name': image_name
                })
        
        print(f"Total training samples: {len(training_data)}")
        return training_data
    
    def simulate_bbox(self, field, index):
        """Simulate bounding box based on field type and position"""
        # In real case, get actual bounding boxes from OCR
        # For now, simulate based on field type
        field_positions = {
            'student_name': (100, 50, 200, 80),
            'student_id': (100, 100, 300, 130),
            'vehicle_plate': (100, 150, 200, 180),
            'instructor_name': (100, 200, 200, 230),
            'distance_completed': (100, 250, 150, 280),
            'time_completed': (200, 250, 250, 280),
            'distance_remaining': (100, 300, 150, 330),
            'time_remaining': (200, 300, 250, 330),
            'total_sessions': (100, 350, 150, 380)
        }
        
        base_bbox = field_positions.get(field, (100, 50, 200, 80))
        # Add some variation
        x1, y1, x2, y2 = base_bbox
        return (x1 + index * 10, y1 + index * 5, x2 + index * 10, y2 + index * 5)
    
    def extract_spatial_features(self, current_item, all_items, current_index):
        """Extract spatial features for current text"""
        features = []
        
        # Current bbox features
        x1, y1, x2, y2 = current_item['bbox']
        features.extend([x1, y1, x2, y2, x2-x1, y2-y1])  # bbox + width, height
        
        # Relative position features
        for i, item in enumerate(all_items):
            if i != current_index:
                other_x1, other_y1, other_x2, other_y2 = item['bbox']
                
                # Distance between centers
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                other_center_x = (other_x1 + other_x2) / 2
                other_center_y = (other_y1 + other_y2) / 2
                
                distance = np.sqrt((center_x - other_center_x)**2 + (center_y - other_center_y)**2)
                features.append(distance)
                
                # Relative position (above, below, left, right)
                features.extend([
                    1 if center_y < other_y1 else 0,  # above
                    1 if center_y > other_y2 else 0,  # below
                    1 if center_x < other_x1 else 0,  # left
                    1 if center_x > other_x2 else 0   # right
                ])
        
        # Pad or truncate to fixed length
        max_features = 50
        if len(features) > max_features:
            features = features[:max_features]
        else:
            features.extend([0] * (max_features - len(features)))
        
        return features
    
    def preprocess_text(self, text):
        """Preprocess text for classification"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self, training_data):
        """Prepare data for training"""
        print("Preparing spatial data...")
        
        # Extract texts and labels
        texts = [self.preprocess_text(item['text']) for item in training_data]
        labels = [item['field'] for item in training_data]
        spatial_features = [item['spatial_features'] for item in training_data]
        
        # Create tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        X_text = self.tokenizer.texts_to_sequences(texts)
        X_text = pad_sequences(X_text, maxlen=self.max_length, padding='post')
        
        # Prepare spatial features
        X_spatial = np.array(spatial_features)
        
        # Normalize spatial features
        self.scaler = StandardScaler()
        X_spatial = self.scaler.fit_transform(X_spatial)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        return X_text, X_spatial, y
    
    def build_spatial_text_model(self, vocab_size, max_length, spatial_dim, num_classes):
        """Build spatial-text combined model"""
        
        # Text input branch
        text_input = Input(shape=(max_length,), name='text_input')
        text_embed = Embedding(vocab_size, 128)(text_input)
        text_lstm = LSTM(64, dropout=0.2)(text_embed)
        text_dense = Dense(32, activation='relu')(text_lstm)
        
        # Spatial input branch
        spatial_input = Input(shape=(spatial_dim,), name='spatial_input')
        spatial_dense1 = Dense(64, activation='relu')(spatial_input)
        spatial_dense2 = Dense(32, activation='relu')(spatial_dense1)
        
        # Combine branches
        combined = Concatenate()([text_dense, spatial_dense2])
        combined_dense = Dense(64, activation='relu')(combined)
        combined_dropout = Dropout(0.3)(combined_dense)
        output = Dense(num_classes, activation='softmax')(combined_dropout)
        
        model = Model(inputs=[text_input, spatial_input], outputs=output)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_text, X_spatial, y):
        """Train the spatial-text model"""
        print("Training spatial-text model...")
        
        # Split data
        X_text_train, X_text_val, X_spatial_train, X_spatial_val, y_train, y_val = train_test_split(
            X_text, X_spatial, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        num_classes = len(np.unique(y))
        spatial_dim = X_spatial.shape[1]
        self.combined_model = self.build_spatial_text_model(
            self.vocab_size, self.max_length, spatial_dim, num_classes
        )
        
        # Train model
        history = self.combined_model.fit(
            [X_text_train, X_spatial_train], y_train,
            validation_data=([X_text_val, X_spatial_val], y_val),
            epochs=30,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def save_model(self):
        """Save trained model and components"""
        print("Saving spatial-text model...")
        
        # Save model
        self.combined_model.save('spatial_text_classifier_model.h5')
        
        # Save components
        with open('spatial_text_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open('spatial_text_labels.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open('spatial_text_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Spatial-text model saved successfully!")
    
    def load_model(self):
        """Load trained model and components"""
        print("Loading spatial-text model...")
        
        # Load model
        self.combined_model = tf.keras.models.load_model('spatial_text_classifier_model.h5')
        
        # Load components
        with open('spatial_text_tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open('spatial_text_labels.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open('spatial_text_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("Spatial-text model loaded successfully!")
    
    def predict_field(self, text, bbox, context_texts=None):
        """Predict field for given text with spatial context"""
        if self.combined_model is None:
            print("Model not loaded!")
            return None
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Extract spatial features
        if context_texts:
            spatial_features = self.extract_spatial_features(
                {'text': text, 'bbox': bbox}, context_texts, 0
            )
        else:
            # Default spatial features if no context
            spatial_features = [0] * 50
        
        spatial_features = self.scaler.transform([spatial_features])
        
        # Predict
        prediction = self.combined_model.predict([padded_sequence, spatial_features], verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        field_name = self.label_encoder.inverse_transform([predicted_class])[0]
        return field_name, confidence

def main():
    """Main function"""
    print("=== SPATIAL TEXT CLASSIFIER ===")
    
    # Initialize classifier
    classifier = SpatialTextClassifier()
    
    # Load data
    training_data = classifier.load_data()
    
    # Prepare data
    X_text, X_spatial, y = classifier.prepare_data(training_data)
    
    # Train model
    history = classifier.train_model(X_text, X_spatial, y)
    
    # Save model
    classifier.save_model()
    
    print("\n✅ Spatial-text training completed!")

if __name__ == "__main__":
    main()
