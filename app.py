from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from functools import lru_cache
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import os
import uuid
import io
import traceback
import json
import base64
import shutil
import cv2
import numpy as np
import pandas as pd
import atexit
import logging

# Import your custom classes - prefer full face recognition
try:
    from processor_full import FaceProcessor
    FACE_RECOGNITION_ENABLED = True
    print("Using full face recognition processor")
except ImportError:
    try:
        from processor import FaceProcessor
        FACE_RECOGNITION_ENABLED = True
        print("Using standard face recognition processor")
    except ImportError as e:
        print(f"Warning: Face recognition not available: {e}")
        from processor_simple import SimpleFaceProcessor as FaceProcessor
        FACE_RECOGNITION_ENABLED = False
        print("Using simplified processor")

from database import Database
from camera import Camera

# Add to app.py - at the top of the file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("smart_presence")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'smart_presence_ai_secret_key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
db = Database()
processor = FaceProcessor('training_data')
camera = Camera.get_instance()

# Connect processor to camera
camera.set_processor(processor)

# Force model training
print("Training face recognition model...")
try:
    processor.train_model()
    print(f"Model trained with {len(processor.known_face_encodings)} faces")
except Exception as e:
    print(f"Error training model: {e}")

app.config['EXPORT_FOLDER'] = 'static/exports'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['EXPORT_FOLDER'], exist_ok=True)
os.makedirs('training_data', exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cache dashboard data for 5 minutes
@lru_cache(maxsize=1)
def get_cached_dashboard_data():
    # Cache expiry logic
    get_cached_dashboard_data.last_update = getattr(
        get_cached_dashboard_data, 'last_update', 
        datetime.now() - timedelta(minutes=10)
    )
    
    # If cache is expired, return None to force refresh
    if datetime.now() - get_cached_dashboard_data.last_update > timedelta(minutes=5):
        get_cached_dashboard_data.last_update = datetime.now()
        get_cached_dashboard_data.cache_clear() # Clear LRU cache
        
    # Your existing dashboard data generation code...
    today = datetime.now().date()
    today_str = today.strftime('%Y-%m-%d')
    
    # Initialize dashboard data with defaults
    dashboard_data = {
        'today_count': 0,
        'on_time_count': 0,
        'late_count': 0,
        'avg_hours': 0,
        'weekly_trend': [],
        'top_attendees': [],
        'recent_people': [],
        'face_count': len(processor.known_face_encodings) if hasattr(processor, 'known_face_encodings') else 0,
        'last_model_update': processor.last_training_time.strftime('%Y-%m-%d %H:%M') if hasattr(processor, 'last_training_time') and processor.last_training_time else 'Never'
    }
    
    try:
        # Today's attendance
        today_attendance = db.get_attendance_by_date(today_str)
        
        # Basic counts
        dashboard_data['today_count'] = len(today_attendance)
        
        # On-time vs late counts
        on_time = [r for r in today_attendance if r.get('time', '00:00:00') <= '09:00:00']
        dashboard_data['on_time_count'] = len(on_time)
        dashboard_data['late_count'] = dashboard_data['today_count'] - dashboard_data['on_time_count']
        
        # Calculate average hours
        if today_attendance:
            total_hours = sum(float(r.get('total_hours', 0)) for r in today_attendance)
            dashboard_data['avg_hours'] = round(total_hours / len(today_attendance), 1) if len(today_attendance) > 0 else 0
        
        # Weekly trend data - get data for each day of the current week
        start_of_week = today - timedelta(days=today.weekday())
        dashboard_data['weekly_trend'] = []
        
        for i in range(7):  # 0=Monday through 6=Sunday
            date = start_of_week + timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            day_name = date.strftime('%a')  # Short day name (Mon, Tue, etc.)
            
            # Get attendance for this day
            day_attendance = db.get_attendance_by_date(date_str)
            
            # Count on-time vs late
            day_on_time = len([r for r in day_attendance if r.get('time', '00:00:00') <= '09:00:00'])
            day_late = len(day_attendance) - day_on_time
            
            dashboard_data['weekly_trend'].append({
                'day': day_name,
                'date': date_str,
                'count': len(day_attendance),
                'on_time': day_on_time,
                'late': day_late
            })
        
        # Get top attendees this month
        current_month = today.month
        current_year = today.year
        
        # Get month data if available
        try:
            monthly_data = db.get_monthly_attendance(current_year, current_month)
            if monthly_data and 'people' in monthly_data:
                # Sort by attendance count (days present)
                sorted_people = sorted(monthly_data['people'], 
                                       key=lambda x: x.get('attendance_count', 0), 
                                       reverse=True)
                dashboard_data['top_attendees'] = sorted_people[:5]  # Top 5
        except Exception as e:
            print(f"Error getting monthly data: {e}")
            
        # Get recent registrations
        try:
            recent = db.get_all_people(limit=5)
            if recent:
                dashboard_data['recent_people'] = recent
        except Exception as e:
            print(f"Error getting recent registrations: {e}")
            
    except Exception as e:
        print(f"Error building dashboard: {e}")
        import traceback
        traceback.print_exc()

    return dashboard_data

@app.route('/')
def index():
    """Main dashboard page"""
    if 'user_id' not in session:
        return render_template('index.html', logged_in=False)
    
    # Check if camera is active
    camera_active = camera.is_active() if hasattr(camera, 'is_active') else True
    
    # Calculate server load (placeholder - you can implement a real measure)
    server_load = 25  # placeholder value
    
    # Get cached or fresh dashboard data
    dashboard_data = get_cached_dashboard_data()
        
    return render_template('index.html', 
                          dashboard=dashboard_data, 
                          logged_in=True,
                          camera_active=camera_active,
                          server_load=server_load)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if db.validate_user(username, password):
            user = db.get_user_by_username(username)
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('index.html', login=True)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form['name']
            role = request.form['role']
            person_id = request.form.get('person_id', str(uuid.uuid4()))
            
            # Create directory for the person
            person_dir = os.path.join('training_data', person_id)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save images
            images_saved = 0
            
            # Handle uploaded images
            if 'images' in request.files:
                files = request.files.getlist('images')
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(person_dir, filename)
                        file.save(file_path)
                        images_saved += 1
                        print(f"Saved image: {file_path}")
            
            # Handle base64 captured images
            if 'captured_images' in request.form:
                captured_images = json.loads(request.form['captured_images'])
                for idx, img_data in enumerate(captured_images):
                    if img_data.startswith('data:image'):
                        # Extract base64 data
                        img_data = img_data.split(',')[1]
                        img_bytes = base64.b64decode(img_data)
                        
                        # Save image
                        file_path = os.path.join(person_dir, f"capture_{idx}.jpg")
                        with open(file_path, 'wb') as f:
                            f.write(img_bytes)
                        images_saved += 1
                        print(f"Saved captured image: {file_path}")
            
            # Add person to database
            if images_saved > 0:
                success = db.add_person(person_id, name, role)
                if success:
                    # Retrain the model - use a separate thread to avoid blocking the main thread
                    def train_in_thread():
                        try:
                            processor.train_model()
                            print(f"Successfully trained model in background thread")
                        except Exception as e:
                            print(f"Error in background training: {e}")
                    
                    # Start background thread for training
                    import threading
                    training_thread = threading.Thread(target=train_in_thread)
                    training_thread.daemon = True  # This ensures the thread won't block app shutdown
                    training_thread.start()
                    
                    flash(f'Successfully registered {name} with {images_saved} images. Model training in progress...', 'success')
                else:
                    flash('Failed to add person to database', 'danger')
                    # Remove images if database insertion failed
                    import shutil
                    if os.path.exists(person_dir):
                        shutil.rmtree(person_dir)
            else:
                flash('No valid images provided', 'danger')
                # Remove empty directory
                if os.path.exists(person_dir):
                    os.rmdir(person_dir)
        
        except Exception as e:
            flash(f'Error during registration: {str(e)}', 'danger')
            print(f"Registration error: {e}")
            traceback.print_exc()
        
        return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/live')
def live():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for face recognition"""
    if 'user_id' not in session:
        return "Not authorized", 401
    
    return camera.generate_frames()

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_api():
    """API endpoint to mark attendance"""
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        data = request.get_json()
        
        if not data or 'person_id' not in data:
            return jsonify({"status": "error", "message": "Missing required data"}), 400
        
        person_id = data.get('person_id')
        confidence = data.get('confidence', 0.7)
        
        # Get person details
        person = db.get_person(person_id)
        if not person:
            return jsonify({"status": "error", "message": "Person not found"}), 404
        
        # Mark attendance
        timestamp = datetime.now()
        success = db.mark_attendance(person_id, timestamp, confidence)
        
        if success:
            logger.info(f"Attendance marked via API for {person['name']} (ID: {person_id})")
            return jsonify({
                "status": "success", 
                "message": "Attendance marked",
                "person": {
                    "id": person_id,
                    "name": person['name'],
                    "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')
                }
            })
        else:
            return jsonify({"status": "error", "message": "Failed to mark attendance"}), 500
            
    except Exception as e:
        logger.error(f"Error in mark_attendance_api: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/attendance')
def attendance():
    """Display attendance records"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get date from query params or use today
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    # Get attendance data for the date
    attendance_data = db.get_attendance_by_date(date)
    
    # Pass database file path for debugging
    db_file = db.db_file
    
    return render_template('attendance.html',
                          selected_date=date,
                          attendance=attendance_data,
                          db_file=db_file)

@app.route('/reports')
def reports():
    """Show attendance reports page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get query parameters
    report_type = request.args.get('type', 'daily')
    start_date = request.args.get('start_date', 
                                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', 
                               datetime.now().strftime('%Y-%m-%d'))
    
    # Get all registered people for the dropdown
    all_people = db.get_all_people()
    
    # Initialize empty structures for all possible report types
    report_data = []
    monthly_data = {'people': [], 'dates': [], 'days_count': 0}
    
    # Process report based on type
    if report_type == 'daily':
        report_data = db.get_daily_attendance(start_date, end_date)
    elif report_type == 'by_person':
        person_id = request.args.get('person_id')
        if person_id:
            person = db.get_person(person_id)
            if person:
                # Convert SQLite Row to dict for template use
                person_dict = {
                    'id': person['id'], 
                    'person_id': person['id'],
                    'name': person['name'], 
                    'role': person['role']
                }
                # FIX HERE - get_person_attendance returns a list, not a dict with 'records' key
                person_report = db.get_person_attendance(person_id, start_date, end_date)
                report_data = {
                    'person': person_dict,
                    'attendance': person_report  # Use the list directly, don't access a 'records' key
                }
    elif report_type == 'monthly':
        try:
            month = int(request.args.get('month', datetime.now().month))
            year = int(request.args.get('year', datetime.now().year))
            monthly_data = db.get_monthly_attendance(year, month)
        except Exception as e:
            print(f"Error getting monthly attendance: {e}")
    
    # Pass all required data to template
    return render_template('reports.html', 
                          report_type=report_type,
                          start_date=start_date,
                          end_date=end_date,
                          report_data=report_data,
                          monthly_data=monthly_data,
                          all_people=all_people,
                          current_month=datetime.now().month,
                          current_year=datetime.now().year)

@app.route('/reports/by_person/<person_id>')
def report_by_person(person_id):
    """Show attendance report for specific person"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    # Get date range from query parameters or use defaults
    start_date = request.args.get('start_date', 
                                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', 
                               datetime.now().strftime('%Y-%m-%d'))
    
    # Get person and their attendance records
    person = db.get_person(person_id)
    if not person:
        flash('Person not found', 'danger')
        return redirect(url_for('reports'))
        
    attendance_records = db.get_attendance_by_person(person_id, start_date, end_date)
    
    # Calculate statistics
    total_days = len(set([record['date'] for record in attendance_records]))
    total_hours = sum([record['total_hours'] for record in attendance_records])
    
    return render_template('report_person.html', 
                          person=person, 
                          records=attendance_records,
                          start_date=start_date,
                          end_date=end_date,
                          total_days=total_days,
                          total_hours=total_hours)

@app.route('/export_report', methods=['POST'])
def export_report():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Not authenticated"})
    
    try:
        report_type = request.form.get('report_type')
        
        if report_type == 'monthly':
            month_year = request.form.get('month_year')
            if not month_year:
                return jsonify({"status": "error", "message": "Month and year required"})
                
            year, month = month_year.split('-')
            monthly_data = db.get_monthly_attendance(int(year), int(month))
            
            # Create DataFrame for monthly report
            data = []
            
            for person in monthly_data['people']:
                row = {
                    'Name': person['name'],
                    'Role': person['role'],
                    'Total Hours': person['total_hours'],
                    'Days Present': person['attendance_count']
                }
                
                # Add dates as columns
                for date in monthly_data['dates']:
                    if date in person['dates']:
                        row[date] = person['dates'][date]['hours']
                    else:
                        row[date] = 0
                        
                data.append(row)
                
        elif report_type == 'person':
            person_id = request.form.get('person_id')
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            
            if not person_id or not start_date or not end_date:
                return jsonify({"status": "error", "message": "Missing required parameters"})
                
            person_data = db.get_person_attendance(person_id, start_date, end_date)
            person = db.get_person(person_id)
            
            # Create DataFrame for person report
            data = []
            
            for record in person_data['records']:
                data.append({
                    'Name': person['name'],
                    'Date': record['date'],
                    'Time In': record['time'],
                    'Last Seen': record['last_seen'],
                    'Duration': record['duration'],
                    'Hours': record['hours']
                })
        else:
            return jsonify({"status": "error", "message": "Unknown report type"})
            
        # Export to CSV
        df = pd.DataFrame(data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        # Create response
        filename = f"report_{report_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        response = Response(output.getvalue(), mimetype='text/csv')
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return response
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Export error: {str(e)}"})

# Add this route to handle missing static files gracefully
@app.route('/static/img/no-camera.jpg')
def fallback_camera_image():
    # Return a simple SVG image as fallback
    svg = '''
    <svg width="640" height="480" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f0f0f0"/>
        <text x="50%" y="50%" font-family="Arial" font-size="24" text-anchor="middle">
            Camera Not Available
        </text>
    </svg>
    '''
    return svg, 200, {'Content-Type': 'image/svg+xml'}

@app.route('/video_metadata')
def video_metadata():
    """Stream face recognition metadata as Server-Sent Events"""
    import time  # Add local import to ensure it's available in the generator function
    
    def generate():
        while True:
            # Get the latest frame and detect faces
            _, detected_faces = camera.get_frame()
            
            # Format the data
            data = {
                "timestamp": time.time(),
                "faces": []
            }
            
            # Process faces and build response
            for face in detected_faces:
                data["faces"].append({
                    "person_id": face["person_id"],
                    "name": face["name"],
                    "confidence": face["confidence"],
                    "box": {
                        "x": face["x"],
                        "y": face["y"],
                        "width": face["width"],
                        "height": face["height"]
                    }
                })
            
            # Yield the data in SSE format
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.2)  # 5 updates per second
            
    return Response(generate(), mimetype="text/event-stream")

@app.route('/debug/face_model')
def debug_face_model():
    """Check the face recognition model status"""
    if 'user_id' not in session:
        flash('You must be logged in to access this page', 'danger')
        return redirect(url_for('login'))
    
    # Detailed model info
    model_info = {
        "encodings_count": len(processor.known_face_encodings),
        "names_count": len(processor.known_face_names),
        "ids_count": len(processor.known_face_ids),
        "sample_names": processor.known_face_names[:5] if processor.known_face_names else [],
        "sample_ids": processor.known_face_ids[:5] if processor.known_face_ids else [],
        "model_file": processor.face_model_path,
        "model_exists": os.path.exists(processor.face_model_path),
        "model_size_kb": os.path.getsize(processor.face_model_path) / 1024 if os.path.exists(processor.face_model_path) else 0
    }
    
    # Force model retrain
    if request.args.get('retrain') == '1':
        try:
            print("Manually triggering model retraining...")
            processor.train_model()
            flash(f"Model retrained with {len(processor.known_face_encodings)} faces", "success")
        except Exception as e:
            print(f"Error retraining model: {e}")
            flash(f"Error retraining model: {str(e)}", "danger")
    
    # CHANGE THIS LINE - use debug_face_model.html instead of debug_model.html
    return render_template('debug_face_model.html', model_info=model_info)

@app.route('/reset_database')
def reset_database():
    """Admin function to reset and recreate the database"""
    if 'user_id' not in session:
        flash('You must be logged in to access this page', 'danger')
        return redirect(url_for('login'))
        
    if not session.get('is_admin', False):
        flash('Admin privileges required', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Instead of deleting the file (which causes lock issues),
        # drop all tables and recreate them
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Drop tables in correct order (respecting foreign keys)
        cursor.execute("DROP TABLE IF EXISTS attendance")
        cursor.execute("DROP TABLE IF EXISTS people")
        cursor.execute("DROP TABLE IF EXISTS users")
        
        conn.commit()
        conn.close()
        
        # Recreate the database
        db.create_tables()
        db.ensure_admin_exists()
        
        # Add a message about resetting model
        flash('Database has been reset successfully. Please register new faces to train the model.', 'success')
        return redirect(url_for('register'))
    except Exception as e:
        flash(f'Error resetting database: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/debug/system')
def debug_system():
    """Debug endpoint to check system state"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"})
    
    # Check model state
    model_info = {
        "faces_count": len(processor.known_face_encodings),
        "names": processor.known_face_names,
        "ids": processor.known_face_ids,
        "model_file_exists": os.path.exists(processor.face_model_path),
        "model_file_size": os.path.getsize(processor.face_model_path) if os.path.exists(processor.face_model_path) else 0
    }
    
    # Check training data
    training_info = {
        "directory_exists": os.path.exists(processor.training_data_path),
        "directories": []
    }
    
    if os.path.exists(processor.training_data_path):
        for person_id in os.listdir(processor.training_data_path):
            person_dir = os.path.join(processor.training_data_path, person_id)
            if os.path.isdir(person_dir):
                person_info = {
                    "id": person_id,
                    "image_count": len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]),
                    "directory": person_dir
                }
                training_info["directories"].append(person_info)
    
    # Check database
    conn = db.get_connection()
    cursor = conn.cursor()
    
    db_info = {"tables": {}}
    
    # Check people table
    cursor.execute("SELECT COUNT(*) FROM people")
    db_info["tables"]["people_count"] = cursor.fetchone()[0]
    
    # Get first few people
    cursor.execute("SELECT id, name, role FROM people LIMIT 5")
    db_info["tables"]["people_sample"] = cursor.fetchall()
    
    # Check attendance table
    cursor.execute("SELECT COUNT(*) FROM attendance")
    db_info["tables"]["attendance_count"] = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        "model": model_info,
        "training_data": training_info,
        "database": db_info
    })

@app.route('/debug/register_test')
def register_test_user():
    """Create a test user for debugging"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"})
    
    # Create test person with timestamp to avoid duplicates
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    person_id = f"test_person_{timestamp}"
    name = f"Test Person {timestamp}"
    role = "Test"
    
    # Create directory
    person_dir = os.path.join('training_data', person_id)
    os.makedirs(person_dir, exist_ok=True)
    
    # Create or download a test image (this is a placeholder, you need to provide actual face images)
    test_image_path = os.path.join(person_dir, "test_face.jpg")
    
    if not os.path.exists(test_image_path):
        # Option 1: Copy a sample image from your project
        # shutil.copy("path/to/sample_face.jpg", test_image_path)
        
        # Option 2: Create a blank image with text
        blank_image = np.ones((300, 300, 3), np.uint8) * 255
        cv2.putText(blank_image, "TEST", (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        cv2.imwrite(test_image_path, blank_image)
    
    # Add to database
    db.add_person(person_id, name, role)
    
    # Retrain model
    processor.train_model()
    
    return jsonify({
        "status": "success", 
        "message": "Test user created and model retrained",
        "face_count": len(processor.known_face_encodings)
    })

@app.route('/cleanup_training_data')
def cleanup_training_data():
    """Remove training data for people not in the database"""
    if 'user_id' not in session:
        flash('You must be logged in to access this page', 'danger')
        return redirect(url_for('login'))
        
    if not session.get('is_admin', False):
        flash('Admin privileges required', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Get all registered people
        registered_people = db.get_all_people()
        registered_ids = {person['id'] for person in registered_people}
        
        # Check training data directory
        training_dir = processor.training_data_path
        removed_dirs = []
        
        for person_id in os.listdir(training_dir):
            person_dir = os.path.join(training_dir, person_id)
            
            if os.path.isdir(person_dir) and person_id not in registered_ids:
                # This directory corresponds to a person not in the database
                import shutil
                shutil.rmtree(person_dir)
                removed_dirs.append(person_id)
        
        # Retrain the model
        processor.train_model()
        
        flash(f'Cleaned up {len(removed_dirs)} unregistered directories. Model retrained.', 'success')
        return redirect(url_for('register'))
    
    except Exception as e:
        flash(f'Error during cleanup: {str(e)}', 'danger')
        return redirect(url_for('register'))

@app.route('/export_attendance')
def export_attendance():
    """Export attendance data in various formats"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    export_format = request.args.get('format', 'csv')
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    # Get attendance data
    attendance_data = db.get_attendance_by_date(date)
    
    if export_format == 'csv':
        # Convert to pandas DataFrame for easy CSV export
        import pandas as pd
        import io
        
        # Prepare data for DataFrame
        data = []
        for record in attendance_data:
            data.append({
                'Name': record['name'],
                'Role': record['role'] if 'role' in record else '',
                'Date': record['date'],
                'Time In': record['time'],
                'Last Seen': record['last_seen'],
                'Duration': record['duration'],
                'Hours': record['total_hours'],
                'Confidence': f"{int(record['confidence'] * 100)}%"
            })
        
        # Create DataFrame and export to CSV
        df = pd.DataFrame(data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        # Create response
        response = Response(output.getvalue(), mimetype='text/csv')
        response.headers["Content-Disposition"] = f"attachment; filename=attendance_{date}.csv"
        return response
    
    elif export_format == 'pdf':
        # For PDF generation you'd typically use a library like ReportLab or WeasyPrint
        # This is a simplified placeholder
        
        return "PDF export not implemented yet", 501
    
    else:
        flash(f"Unsupported export format: {export_format}", "danger")
        return redirect(url_for('attendance'))

@app.route('/health')
def health_check():
    """Health check endpoint for Render monitoring"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "unknown",
            "face_processor": "unknown"
        }
        
        # Test database connection
        try:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            health_status["database"] = "connected"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
            
        # Test face processor
        try:
            if hasattr(processor, 'known_face_encodings'):
                health_status["face_processor"] = f"loaded ({len(processor.known_face_encodings)} faces)"
            else:
                health_status["face_processor"] = "loaded (simple mode)"
        except Exception as e:
            health_status["face_processor"] = f"error: {str(e)}"
            
        return jsonify(health_status), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/debug/dashboard')
def debug_dashboard():
    """Debug endpoint to check dashboard data"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"})
    
    # Get today's date
    today = datetime.now().date()
    today_str = today.strftime('%Y-%m-%d')
    
    # Collect debug info
    debug_info = {
        "today": today_str,
        "attendance": []
    }
    
    # Check today's attendance
    try:
        today_attendance = db.get_attendance_by_date(today_str)
        debug_info["attendance"] = today_attendance
        debug_info["attendance_count"] = len(today_attendance)
    except Exception as e:
        debug_info["attendance_error"] = str(e)
    
    # Check people table
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM people")
        result = cursor.fetchone()
        debug_info["people_count"] = result["count"] if result else 0
        conn.close()
    except Exception as e:
        debug_info["people_error"] = str(e)
    
    # Check face recognition model
    try:
        debug_info["face_count"] = len(processor.known_face_encodings)
        debug_info["face_ids"] = processor.known_face_ids
        debug_info["face_names"] = processor.known_face_names
    except Exception as e:
        debug_info["face_error"] = str(e)
    
    return jsonify(debug_info)

@app.route('/batch_attendance', methods=['GET', 'POST'])
def batch_attendance():
    """Process batch attendance from uploaded image"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    results = []
    today_date = datetime.now().strftime('%Y-%m-%d')
    processed_image = None
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['image']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        date_str = request.form.get('date', today_date)
        
        if file and allowed_file(file.filename):
            try:
                # Read image
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    flash('Invalid image format', 'danger')
                    return redirect(request.url)
                
                # Process image to detect faces
                processed_img, detected_faces = processor.process_image_for_attendance(img)
                
                # Mark attendance for each detected face
                timestamp = datetime.strptime(f"{date_str} {datetime.now().strftime('%H:%M:%S')}", 
                                           '%Y-%m-%d %H:%M:%S')
                
                for face in detected_faces:
                    person_id = face.get('person_id', None)
                    name = face.get('name', 'Unknown')
                    confidence = face.get('confidence', 0)
                    
                    if person_id and confidence > 0.5:  # Only mark attendance if confidence is reasonable
                        db.mark_attendance(person_id, timestamp, confidence)
                        results.append({
                            'person_id': person_id,
                            'name': name,
                            'confidence': confidence,
                            'status': 'Marked'
                        })
                    else:
                        results.append({
                            'person_id': None,
                            'name': 'Unknown',
                            'confidence': confidence if confidence else 0,
                            'status': 'Not recognized'
                        })
                
                # Save the processed image with face boxes for display
                _, buffer = cv2.imencode('.jpg', processed_img)
                processed_image = base64.b64encode(buffer).decode('utf-8')
                
                flash(f'Successfully processed image and marked attendance for {len([r for r in results if r["status"] == "Marked"])} people', 'success')
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                traceback.print_exc()
    
    return render_template('batch_attendance.html', 
                          results=results, 
                          processed_image=processed_image,
                          today_date=today_date)

def shutdown_app():
    """Clean up resources on app shutdown"""
    print("Shutting down application...")
    try:
        if 'camera' in globals():
            print("Releasing camera resources...")
            camera.release()
            print("Camera resources released")
    except Exception as e:
        print(f"Error during shutdown: {e}")
        import traceback
        traceback.print_exc()

def ensure_test_data():
    """Make sure there's at least some test data in the system"""
    if not db.get_all_people():
        print("No people found in database, adding a test person")
        test_id = "TEST123456"
        test_name = "Test Person"
        test_role = "Developer"
        db.add_person(test_id, test_name, test_role)
        print(f"Added test person: {test_name}")

# Call this function when the app starts
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    ensure_test_data()
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
    # Register shutdown handler
    atexit.register(shutdown_app)