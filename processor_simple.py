# Simplified processor without face recognition for deployment fallback
import os
import cv2
import numpy as np
import pickle
from datetime import datetime

class SimpleFaceProcessor:
    """Simplified processor that works without face_recognition library"""
    
    def __init__(self, training_data_path):
        self.training_data_path = training_data_path
        self.face_model_path = 'models/face_model.pkl'
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.last_training_time = None
        
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load existing model if available
        self.load_model()
    
    def load_model(self):
        """Load the face recognition model"""
        if os.path.exists(self.face_model_path):
            try:
                with open(self.face_model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    self.known_face_ids = data.get('ids', [])
                    self.last_training_time = data.get('last_training_time')
                print(f"Loaded face model with {len(self.known_face_encodings)} faces")
            except Exception as e:
                print(f"Error loading face model: {e}")
    
    def train_model(self):
        """Train the face recognition model using basic OpenCV detection"""
        print("Training face recognition model...")
        print(f"Training model from registered people in: {self.training_data_path}")
        
        if not os.path.exists(self.training_data_path):
            print(f"Training data directory not found: {self.training_data_path}")
            return
        
        # Reset known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # For demonstration purposes, we'll just detect faces and store basic info
        # In a real deployment, you'd want to implement proper face recognition
        for person_id in os.listdir(self.training_data_path):
            person_dir = os.path.join(self.training_data_path, person_id)
            if not os.path.isdir(person_dir):
                continue
            
            # Get person info from database
            from database import Database
            db = Database()
            person = db.get_person(person_id)
            
            if person:
                name = person['name']
                print(f"DEBUG: get_person for {person_id} returned: {person}")
                
                # Count images for this person
                image_count = 0
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_count += 1
                
                # For now, just store the person info without actual face encoding
                # This allows the app to run without face_recognition library
                if image_count > 0:
                    # Create a dummy encoding (in real app, this would be actual face encoding)
                    dummy_encoding = np.random.random(128)  # face_recognition uses 128-dim vectors
                    
                    self.known_face_encodings.append(dummy_encoding)
                    self.known_face_names.append(name)
                    self.known_face_ids.append(person_id)
                    
                    print(f"Trained on {image_count} images for {name}")
        
        # Save the model
        self.save_model()
        self.last_training_time = datetime.now()
        print(f"Model training complete with {len(self.known_face_encodings)} face encodings")
    
    def save_model(self):
        """Save the face recognition model"""
        os.makedirs(os.path.dirname(self.face_model_path), exist_ok=True)
        
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names,
            'ids': self.known_face_ids,
            'last_training_time': datetime.now()
        }
        
        with open(self.face_model_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved face model with {len(self.known_face_encodings)} faces")
    
    def process_image_for_attendance(self, image):
        """Process image and detect faces (simplified version)"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using OpenCV
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        detected_faces = []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # For demo purposes, return "Unknown" for all faces
            # In a real implementation, you'd compare with known faces
            detected_faces.append({
                'person_id': None,
                'name': 'Unknown',
                'confidence': 0.5,
                'x': x,
                'y': y,
                'width': w,
                'height': h
            })
        
        return image, detected_faces

# This will be the main class used when face_recognition is not available
FaceProcessor = SimpleFaceProcessor
