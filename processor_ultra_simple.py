"""
Ultra-simple processor for environments where face recognition libraries fail to install
This processor provides basic camera functionality without any face recognition
"""

import os
import time
from datetime import datetime
import json

class UltraSimpleProcessor:
    """Ultra-simple processor that provides basic functionality without any ML dependencies"""
    
    def __init__(self, training_data_path='training_data'):
        self.training_data_path = training_data_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.face_model_path = os.path.join('models', 'simple_model.json')
        self.last_training_time = datetime.now()
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs(training_data_path, exist_ok=True)
        
        print("UltraSimpleProcessor initialized - no face recognition capabilities")
    
    def train_model(self):
        """Simulate model training by counting registered people"""
        try:
            people_count = 0
            if os.path.exists(self.training_data_path):
                people_count = len([d for d in os.listdir(self.training_data_path) 
                                 if os.path.isdir(os.path.join(self.training_data_path, d))])
            
            # Create mock encodings
            self.known_face_encodings = [f"encoding_{i}" for i in range(people_count)]
            self.known_face_names = [f"Person_{i}" for i in range(people_count)]
            self.known_face_ids = [f"id_{i}" for i in range(people_count)]
            
            # Save simple model info
            model_data = {
                'people_count': people_count,
                'last_training': datetime.now().isoformat(),
                'processor_type': 'ultra_simple'
            }
            
            with open(self.face_model_path, 'w') as f:
                json.dump(model_data, f)
            
            self.last_training_time = datetime.now()
            print(f"Simple model updated: {people_count} people registered")
            
        except Exception as e:
            print(f"Error in simple model training: {e}")
    
    def process_frame(self, frame):
        """Process frame without any face recognition"""
        # Return empty list since we can't detect faces
        return frame, []
    
    def process_image_for_attendance(self, image):
        """Process image for batch attendance - returns empty results"""
        # Since we can't detect faces, return the original image and no detections
        return image, []
    
    def recognize_face(self, face_encoding):
        """Mock face recognition - always returns unknown"""
        return "Unknown", 0.0, None
    
    def save_model(self):
        """Save the simple model"""
        try:
            model_data = {
                'people_count': len(self.known_face_encodings),
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                'processor_type': 'ultra_simple'
            }
            
            with open(self.face_model_path, 'w') as f:
                json.dump(model_data, f)
        except Exception as e:
            print(f"Error saving simple model: {e}")
    
    def load_model(self):
        """Load the simple model"""
        try:
            if os.path.exists(self.face_model_path):
                with open(self.face_model_path, 'r') as f:
                    model_data = json.load(f)
                    print(f"Loaded simple model: {model_data}")
        except Exception as e:
            print(f"Error loading simple model: {e}")

# For compatibility, alias the main class
SimpleFaceProcessor = UltraSimpleProcessor
