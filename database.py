import sqlite3
import os
import hashlib
from datetime import datetime, timedelta

class Database:
    def __init__(self, db_file='smart_presence.db'):
        self.db_file = db_file
        self.create_tables()
        self.ensure_admin_exists()
    
    def get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def create_tables(self):
        """Create necessary database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create tables with consistent schema
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            date TEXT,
            time TEXT,
            last_seen TEXT,
            confidence REAL DEFAULT 0,
            verified INTEGER DEFAULT 0,
            FOREIGN KEY (person_id) REFERENCES people (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        print("Database tables created successfully")
    
    def ensure_admin_exists(self):
        """Ensure at least one admin user exists"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if any admin exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Create default admin user
            default_username = "admin"
            default_password = "admin123"
            password_hash = hashlib.sha256(default_password.encode()).hexdigest()
            
            cursor.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
                (default_username, password_hash)
            )
            print("Created default admin user 'admin' with password 'admin123'")
        
        conn.commit()
        conn.close()
    
    def add_user(self, username, password, is_admin=False):
        """Add a new user to the database"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Make sure we're using password_hash column consistently
            cursor.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                (username, password_hash, is_admin)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Username already exists
            return False
        finally:
            conn.close()
    
    def validate_user(self, username, password):
        """Validate user credentials"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        user = cursor.fetchone()
        conn.close()
        
        return user is not None
    
    def get_user_by_username(self, username):
        """Get user details by username"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        return dict(user) if user else None
    
    def add_person(self, person_id, name, role=None):
        """Add a new person to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO people (id, name, role) VALUES (?, ?, ?)",
                (person_id, name, role)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Person ID already exists
            return False
        finally:
            conn.close()
    
    def get_person(self, person_id):
        """Get person details by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM people WHERE id = ?", (person_id,))
        person = cursor.fetchone()
        
        conn.close()
        
        print(f"DEBUG: get_person for {person_id} returned: {person}")
        
        return person
    
    def get_all_people(self, limit=None):
        """Get all registered people"""
        try:
            conn = self.get_connection()  # Get a new connection
            cursor = conn.cursor()
            
            # First check total count
            cursor.execute("SELECT COUNT(*) as count FROM people")
            count_row = cursor.fetchone()
            total_count = count_row['count'] if count_row else 0
            print(f"DEBUG: Total people in database: {total_count}")
            
            # Modified query - remove created_at column which doesn't exist
            if limit:
                cursor.execute("""
                    SELECT id, name, role
                    FROM people
                    ORDER BY name
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id, name, role
                    FROM people
                    ORDER BY name
                """)
                    
            # Check row count
            rows = cursor.fetchall()
            print(f"DEBUG: get_all_people found {len(rows)} people records")
            
            results = []
            # Then convert each row to a dictionary
            for row in rows:
                person = {
                    'id': row['id'],
                    'person_id': row['id'],
                    'name': row['name'],
                    'role': row['role'],
                    'created_at': datetime.now().strftime('%Y-%m-%d')  # Default value since column doesn't exist
                }
                results.append(person)
            
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting all people: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def mark_attendance(self, person_id, timestamp, confidence=0):
        """Mark attendance for a person"""
        print(f"DEBUG: Marking attendance for {person_id} at {timestamp} with confidence {confidence}")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Format the timestamp
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H:%M:%S')
        
        try:
            # Check if already marked for today - Using proper SQL placeholders
            cursor.execute(
                "SELECT * FROM attendance WHERE person_id = ? AND date = ?", 
                (person_id, date_str)
            )
            existing = cursor.fetchone()
            
            if existing:
                print(f"DEBUG: Already marked attendance for {person_id} on {date_str}")
                # Update last seen time
                cursor.execute(
                    "UPDATE attendance SET last_seen = ?, confidence = ? WHERE person_id = ? AND date = ?",
                    (time_str, max(confidence, existing['confidence']), person_id, date_str)
                )
                conn.commit()
                print(f"DEBUG: Updated last seen time for {person_id} to {time_str}")
            else:
                # Insert new attendance record
                print(f"DEBUG: Inserting new attendance record for {person_id}")
                cursor.execute(
                    "INSERT INTO attendance (person_id, date, time, last_seen, confidence) VALUES (?, ?, ?, ?, ?)",
                    (person_id, date_str, time_str, time_str, confidence)
                )
                conn.commit()
                print(f"DEBUG: New attendance record for {person_id} on {date_str} at {time_str}")
        except Exception as e:
            print(f"ERROR in mark_attendance: {e}")
        finally:
            conn.close()
    
    def get_attendance_by_date(self, date_str):
        """Get attendance records for a specific date"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # If date_str is empty, use today's date
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # For debugging
        print(f"DEBUG: Getting attendance for date: {date_str}")
        
        try:
            cursor.execute("""
                SELECT a.id, a.person_id, p.name, p.role, a.date, a.time, a.last_seen, a.confidence
                FROM attendance a
                JOIN people p ON a.person_id = p.id
                WHERE a.date = ?
                ORDER BY a.time
            """, (date_str,))
            
            records = []
            for row in cursor.fetchall():
                try:
                    # Calculate time duration
                    start_time = datetime.strptime(row['time'], '%H:%M:%S')
                    end_time = datetime.strptime(row['last_seen'], '%H:%M:%S')
                    duration = end_time - start_time
                    total_seconds = duration.total_seconds()
                    hours = total_seconds / 3600  # Convert to hours
                    
                    records.append({
                        'id': row['id'],
                        'person_id': row['person_id'],
                        'name': row['name'],
                        'role': row['role'],
                        'date': row['date'],
                        'time': row['time'],
                        'last_seen': row['last_seen'],
                        'duration': f"{int(total_seconds // 3600)}h {int((total_seconds % 3600) // 60)}m",
                        'total_hours': round(hours, 2),
                        'confidence': row['confidence']
                    })
                except Exception as e:
                    print(f"ERROR processing attendance record: {e}")
                    # Add a minimal record with available data
                    records.append({
                        'id': row['id'] if 'id' in row else 'unknown',
                        'person_id': row['person_id'] if 'person_id' in row else 'unknown',
                        'name': row['name'] if 'name' in row else 'Unknown',
                        'role': row['role'] if 'role' in row else '',
                        'date': row['date'] if 'date' in row else date_str,
                        'time': row['time'] if 'time' in row else '00:00:00',
                        'last_seen': row['last_seen'] if 'last_seen' in row else '00:00:00',
                        'duration': '0h 0m',
                        'total_hours': 0,
                        'confidence': 0
                    })
            
            print(f"DEBUG: Found {len(records)} attendance records for {date_str}")
            
        except Exception as e:
            print(f"ERROR in get_attendance_by_date: {e}")
            records = []
        
        conn.close()
        return records
    
    def get_attendance_report(self, start_date, end_date, person_id=None):
        """Generate attendance report for a date range and optional person"""
        start_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_obj = datetime.strptime(end_date, '%Y-%m-%d')
        end_obj = end_obj.replace(hour=23, minute=59, second=59)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT a.*, p.name, p.role,
               strftime('%Y-%m-%d', a.timestamp) as date
        FROM attendance a
        JOIN people p ON a.person_id = p.id
        WHERE a.timestamp BETWEEN ? AND ?
        """
        
        params = [start_obj, end_obj]
        
        if person_id:
            query += " AND a.person_id = ?"
            params.append(person_id)
        
        query += " ORDER BY a.timestamp"
        
        cursor.execute(query, params)
        attendance_records = [dict(row) for row in cursor.fetchall()]
        
        # Group by date and person
        report_data = []
        for record in attendance_records:
            date = record['date']
            person_id = record['person_id']
            
            # Check if this person already has an entry for this date
            existing = next((item for item in report_data 
                          if item['date'] == date and item['person_id'] == person_id), None)
            
            if existing:
                # Update first/last times if needed
                record_time = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                existing_first = datetime.strptime(existing['first_time'], '%H:%M:%S')
                existing_last = datetime.strptime(existing['last_time'], '%H:%M:%S')
                
                record_time_str = record_time.strftime('%H:%M:%S')
                
                if record_time.time() < existing_first.time():
                    existing['first_time'] = record_time_str
                
                if record_time.time() > existing_last.time():
                    existing['last_time'] = record_time_str
            else:
                # Create new entry
                record_time = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                time_str = record_time.strftime('%H:%M:%S')
                
                report_data.append({
                    'date': date,
                    'person_id': person_id,
                    'name': record['name'],
                    'role': record['role'],
                    'first_time': time_str,
                    'last_time': time_str,
                    'total_hours': 0  # Will be calculated at the end
                })
        
        # Calculate total hours
        for entry in report_data:
            first_time = datetime.strptime(entry['first_time'], '%H:%M:%S')
            last_time = datetime.strptime(entry['last_time'], '%H:%M:%S')
            
            # Calculate difference in hours
            diff = last_time - first_time
            hours = diff.total_seconds() / 3600
            entry['total_hours'] = round(hours, 2)
        
        conn.close()
        return report_data
    
    def verify_attendance(self, attendance_id):
        """Mark an attendance record as verified"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE attendance SET verified = 1 WHERE id = ?",
            (attendance_id,)
        )
        conn.commit()
        conn.close()
    
    def initialize_database(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        
        # People table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT,
                date TEXT,
                time TEXT,
                last_seen TEXT,
                confidence REAL DEFAULT 0,
                FOREIGN KEY (person_id) REFERENCES people (id)
            )
        """)
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        """)
        
        # Create default admin user if not exists
        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                ('admin', 'admin123', 1)  # In production, use hashed password
            )
        
        conn.commit()
        conn.close()
        
        print("Database initialized successfully")
    
    def get_monthly_attendance(self, year, month):
        """Get attendance summary for a month"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get first and last day of month
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{last_day}"
        
        try:
            # Get days with attendance in this month
            cursor.execute("""
                SELECT DISTINCT date FROM attendance
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            """, (start_date, end_date))
            
            dates = [row['date'] for row in cursor.fetchall()]
            
            # Get all people with attendance in this month
            cursor.execute("""
                SELECT DISTINCT p.id, p.name, p.role
                FROM attendance a
                JOIN people p ON a.person_id = p.id
                WHERE a.date BETWEEN ? AND ?
                ORDER BY p.name
            """, (start_date, end_date))
            
            people = cursor.fetchall()
            
            # Build attendance matrix
            attendance_data = []
            
            for person in people:
                # Get attendance for each date
                cursor.execute("""
                    SELECT date, time, last_seen, confidence
                    FROM attendance
                    WHERE person_id = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                """, (person['id'], start_date, end_date))
                
                attendance_dates = {}
                total_hours = 0
                
                for att in cursor.fetchall():
                    try:
                        # Calculate hours
                        start_time = datetime.strptime(att['time'], '%H:%M:%S')
                        end_time = datetime.strptime(att['last_seen'], '%H:%M:%S')
                        duration = end_time - start_time
                        hours = duration.total_seconds() / 3600
                        total_hours += hours
                        
                        attendance_dates[att['date']] = {
                            'time': att['time'],
                            'last_seen': att['last_seen'],
                            'hours': round(hours, 2),
                            'confidence': att['confidence']
                        }
                    except Exception as e:
                        print(f"Error calculating duration for {person['name']} on {att['date']}: {e}")
                
                attendance_data.append({
                    'id': person['id'],
                    'name': person['name'],
                    'role': person['role'],
                    'dates': attendance_dates,
                    'total_hours': round(total_hours, 2),
                    'attendance_count': len(attendance_dates)
                })
            
            return {
                'dates': dates,
                'people': attendance_data,
                'days_count': len(dates)
            }
            
        except Exception as e:
            print(f"Error in get_monthly_attendance: {e}")
            return {'dates': [], 'people': [], 'days_count': 0}
        finally:
            conn.close()
    
    def get_person_attendance(self, person_id, start_date, end_date):
        """Get attendance records for a specific person in a date range"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    date as date_str,
                    time as first_seen,
                    last_seen,
                    (strftime('%s', last_seen) - strftime('%s', time)) / 3600.0 as total_hours
                FROM attendance
                WHERE person_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (person_id, start_date, end_date))
            
            records = []
            for row in cursor.fetchall():
                try:
                    hours = row['total_hours'] if row['total_hours'] else 0
                    minutes = (hours - int(hours)) * 60
                    
                    records.append({
                        'date': row['date_str'],
                        'time': row['first_seen'],
                        'last_seen': row['last_seen'],
                        'duration': f"{int(hours)}h {int(minutes)}m",
                        'total_hours': round(hours, 2)
                    })
                except Exception as e:
                    print(f"Error processing attendance record: {e}")
            
            conn.close()
            return records
        except Exception as e:
            print(f"Error in get_attendance_by_person: {e}")
            return []
    
    def get_daily_attendance(self, start_date, end_date):
        """Get daily attendance summary between two dates"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Query to get attendance counts by date
            cursor.execute("""
                SELECT 
                    date as date_str,
                    COUNT(DISTINCT person_id) as total_count,
                    SUM(CASE WHEN time <= '09:00:00' THEN 1 ELSE 0 END) as on_time,
                    SUM(CASE WHEN time > '09:00:00' THEN 1 ELSE 0 END) as late
                FROM (
                    SELECT 
                        person_id,
                        date,
                        MIN(time) as time
                    FROM attendance
                    WHERE date BETWEEN ? AND ?
                    GROUP BY person_id, date
                )
                GROUP BY date
                ORDER BY date
            """, (start_date, end_date))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'date': row['date_str'],
                    'count': row['total_count'], 
                    'on_time': row['on_time'] or 0,  # Handle NULL values
                    'late': row['late'] or 0         # Handle NULL values
                })
            
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting daily attendance: {e}")
            import traceback
            traceback.print_exc()
            return []