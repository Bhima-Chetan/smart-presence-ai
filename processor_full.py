import os
import cv2
import numpy as np
import pickle
import face_recognition
from datetime import datetime
import time
import json
import threading

class FaceProcessor:
    def __init__(self, training_data_path='training_data', face_model_path='face_model.pkl'):
        # Add a lock for thread-safe access
        self.model_lock = threading.RLock()
        self.training_data_path = training_data_path
        self.face_model_path = face_model_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.last_training_time = None
        
        # Create training directory if it doesn't exist
        os.makedirs(training_data_path, exist_ok=True)
        
        # Load existing model if available
        self.load_model()
    
    def load_model(self):
        """Load face recognition model"""
        try:
            if os.path.exists(self.face_model_path):
                with open(self.face_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.known_face_encodings = model_data.get('encodings', [])
                    self.known_face_ids = model_data.get('ids', [])
                    self.known_face_names = model_data.get('names', [])
                    self.last_training_time = model_data.get('timestamp', datetime.now())
                print(f"Loaded face model with {len(self.known_face_encodings)} faces")
            else:
                print("No existing face model found, will train on first use")
        except Exception as e:
            print(f"Error loading face model: {e}")
            self.known_face_encodings = []
            self.known_face_ids = []
            self.known_face_names = []
    
    def save_model(self):
        """Save face recognition model"""
        try:
            model_data = {
                'encodings': self.known_face_encodings,
                'ids': self.known_face_ids, 
                'names': self.known_face_names,
                'timestamp': datetime.now()
            }
            with open(self.face_model_path, 'wb') as f:
                pickle.dump(model_data, f)
            self.last_training_time = datetime.now()
            print(f"Saved face model with {len(self.known_face_encodings)} faces")
        except Exception as e:
            print(f"Error saving face model: {e}")
    
    def train_model(self):
        """Train the facial recognition model from the training data directory"""
        try:
            # Import here to avoid circular import
            from database import Database
            
            print(f"Training model from registered people in: {self.training_data_path}")
            
            # Create temporary storage for new training data
            temp_encodings = []
            temp_names = []
            temp_ids = []
            
            # Connect to database
            db = Database()
            
            # Iterate through each person's directory
            for person_id in os.listdir(self.training_data_path):
                person_dir = os.path.join(self.training_data_path, person_id)
                
                if not os.path.isdir(person_dir):
                    continue
                
                # Get person's name from database
                person = db.get_person(person_id)
                if not person:
                    print(f"Warning: Person with ID {person_id} not found in database")
                    continue
                
                name = person["name"]
                
                # Process each image in the directory
                image_count = 0
                image_files = [f for f in os.listdir(person_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for image_file in image_files:
                    try:
                        # Load image
                        image_path = os.path.join(person_dir, image_file)
                        image = face_recognition.load_image_file(image_path)
                        
                        # Find face locations
                        face_locations = face_recognition.face_locations(image)
                        
                        if not face_locations:
                            print(f"No face found in {image_path}")
                            continue
                        
                        # Use the first face in the image
                        encoding = face_recognition.face_encodings(image, face_locations)[0]
                        
                        # Add to temporary storage
                        temp_encodings.append(encoding)
                        temp_names.append(name)
                        temp_ids.append(person_id)
                        
                        image_count += 1
                    except Exception as e:
                        print(f"Error processing image {image_file}: {e}")
                
                print(f"Trained on {image_count} images for {name}")
            
            # Now update the model data atomically
            with self.model_lock:
                self.known_face_encodings = temp_encodings
                self.known_face_names = temp_names
                self.known_face_ids = temp_ids
                self.last_training_time = datetime.now()
            
            # Save the model
            self.save_model()
            print(f"Model training complete with {len(self.known_face_encodings)} face encodings")
            
            return True
        except Exception as e:
            print(f"Error in train_model: {e}")
            # Ensure we have at least empty arrays
            with self.model_lock:
                self.known_face_encodings = self.known_face_encodings or []
                self.known_face_names = self.known_face_names or []
                self.known_face_ids = self.known_face_ids or []
            return False
    
    def process_frame(self, frame):
        """Process a video frame to detect and recognize faces"""
        # Make a copy to avoid modifying the original
        display_frame = frame.copy()
        
        # Reduce size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert to RGB (face_recognition uses RGB)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        detected_faces = []
        
        # Process each face - use the lock to safely access the model data
        with self.model_lock:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # See if the face is a match for known faces
                matches = []
                name = "Unknown"
                person_id = None
                confidence = 0
                
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    
                    if True in matches:
                        # Find the best match
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            person_id = self.known_face_ids[best_match_index]
                
                # Draw a box around the face
                color = (0, 255, 0) if confidence >= 0.6 else (0, 0, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                
                # Draw a label with the name below the face
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(display_frame, f"{name} ({confidence:.2f})", (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Add to detected faces
                detected_faces.append({
                    'person_id': person_id,
                    'name': name,
                    'confidence': float(confidence),
                    'x': left,
                    'y': top, 
                    'width': right - left,
                    'height': bottom - top
                })
        
        return display_frame, detected_faces
    
    def process_image_for_attendance(self, image):
        """Process an image to detect and recognize faces for attendance"""
        # Make a copy to avoid modifying the original
        frame = image.copy()
        
        # Reduce size for faster processing if the image is too large
        max_size = 800
        h, w = frame.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return frame, []
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        detected_faces = []
        
        # Match with known faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = []
            if self.known_face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            
            name = "Unknown"
            person_id = None
            confidence = 0.0
            
            if True in matches:
                # Find best match
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                if matches[best_match_index]:
                    person_id = self.known_face_ids[best_match_index]
                    name = self.known_face_names[best_match_index]
            
            # Draw rectangle and label
            color = (0, 255, 0) if confidence >= 0.6 else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Label with name and confidence
            conf_text = f"{confidence:.1%}"
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, conf_text, (left + 6, bottom + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add to results
            detected_faces.append({
                "person_id": person_id,
                "name": name,
                "confidence": float(confidence),
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top
            })
        
        return frame, detected_faces