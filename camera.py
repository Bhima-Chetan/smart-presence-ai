import cv2
import time
import threading
import atexit
import numpy as np
import json
from flask import Response

class Camera:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Camera()
        return cls._instance
    
    def __init__(self):
        self.video_capture = None
        self.processor = None
        self.frame = None
        self.detected_faces = []
        self.raw_frame = None
        self.is_running = False
        self.thread = None
        self.lock = threading.RLock()  # Use RLock for nested locking
        self.shutdown_event = threading.Event()  # Add event for clean shutdown
    
    def set_processor(self, processor):
        """Set the face processor for this camera"""
        self.processor = processor
        
    def initialize(self):
        """Initialize the camera"""
        if self.video_capture is None:
            try:
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    print("Failed to open camera")
                    self.video_capture = None
                    return False
                
                # Start background thread for frame capturing
                if not self.is_running:
                    self.is_running = True
                    self.shutdown_event.clear()  # Clear the shutdown event
                    self.thread = threading.Thread(target=self._update_frame)
                    self.thread.daemon = True
                    self.thread.start()
                    print("Camera processing thread started")
            except Exception as e:
                print(f"Error initializing camera: {e}")
                self.video_capture = None
                return False
        return True
    
    def _update_frame(self):
        """Background thread that continuously captures frames"""
        import time
        from datetime import datetime
        
        # Track when we last marked attendance for each person
        last_attendance_time = {}
        
        # Create a database connection for this thread
        from database import Database
        db = Database()
        
        # Add frame skipping:
        frame_count = 0
        process_every_n_frames = 2  # Process every second frame
        
        try:
            while not self.shutdown_event.is_set() and self.is_running:
                if self.video_capture is None or not self.video_capture.isOpened():
                    print("Camera disconnected - stopping frame capture thread")
                    self.is_running = False
                    break
                    
                try:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        print("Failed to get frame from camera")
                        time.sleep(0.1)
                        continue
                    
                    # Store the raw frame immediately
                    with self.lock:
                        self.raw_frame = frame.copy()
                    
                    # Process the frame if processor is available
                    if self.processor:
                        try:
                            # Use the processor's thread-safe methods
                            processed_frame, detected_faces = self.processor.process_frame(frame)
                            
                            # Debug info for detected faces
                            if detected_faces:
                                print(f"Detected {len(detected_faces)} faces")
                            
                            # Auto-mark attendance for recognized faces with high confidence
                            current_time = time.time()
                            for face in detected_faces:
                                person_id = face.get('person_id')
                                confidence = face.get('confidence', 0)
                                
                                # Only mark with good confidence
                                if person_id and confidence > 0.6:
                                    # Only once per minute
                                    if person_id not in last_attendance_time or \
                                      (current_time - last_attendance_time[person_id]) > 60:
                                        
                                        try:
                                            # Mark attendance
                                            timestamp = datetime.now()
                                            db.mark_attendance(person_id, timestamp, confidence)
                                            
                                            last_attendance_time[person_id] = current_time
                                            print(f"Auto-marked attendance for {face.get('name')} ({confidence:.2f})")
                                        except Exception as e:
                                            print(f"Error auto-marking attendance: {e}")
                            
                            with self.lock:
                                self.frame = processed_frame
                                self.detected_faces = detected_faces
                        except Exception as e:
                            print(f"Error processing frame: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        # No processor available, just store the raw frame
                        with self.lock:
                            self.frame = frame
                            self.detected_faces = []
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Skip processing but still sleep
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    time.sleep(0.03)
                    continue
        except Exception as e:
            print(f"Fatal error in camera thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Camera thread stopping")
    
    def get_frame(self):
        """Get the latest frame and detected faces"""
        if not self.is_running:
            success = self.initialize()
            if not success:
                # If initialization fails, return a blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Camera not available", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', blank_frame)
                return jpeg.tobytes(), []
        
        with self.lock:
            if self.frame is None:
                # If no frame is available, return a blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Waiting for camera", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', blank_frame)
                return jpeg.tobytes(), []
            
            # Copy the frame and faces to avoid race conditions
            frame_copy = self.frame.copy()
            faces_copy = self.detected_faces.copy()
        
        # Convert the frame to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame_copy)
        return jpeg.tobytes(), faces_copy
    
    def generate_frames(self):
        """Generator function for streaming video frames"""
        def gen():
            # Rate limiting
            import time
            last_frame_time = 0
            target_fps = 15
            frame_interval = 1.0 / target_fps
            
            try:
                while True:
                    # Rate limiting
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                        continue
                        
                    last_frame_time = current_time
                    
                    frame_bytes, _ = self.get_frame()
                    
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except GeneratorExit:
                print("Client disconnected from video stream")
            except Exception as e:
                print(f"Error in frame generator: {e}")
                
        return Response(gen(), 
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def is_active(self):
        """Check if the camera is active"""
        return self.is_running and self.video_capture is not None and self.video_capture.isOpened()
    
    def release(self):
        """Release camera resources properly"""
        print("Shutting down camera...")
        
        # Signal the thread to stop
        self.is_running = False
        if hasattr(self, 'shutdown_event'):
            self.shutdown_event.set()
        
        # Wait for the thread to finish (with timeout)
        if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=1.0)  # Wait up to 1 second
                print("Camera thread joined successfully")
            except Exception as e:
                print(f"Error joining camera thread: {e}")
        
        # Release the video capture
        if self.video_capture is not None:
            try:
                self.video_capture.release()
                print("Video capture released")
            except Exception as e:
                print(f"Error releasing video capture: {e}")
            self.video_capture = None