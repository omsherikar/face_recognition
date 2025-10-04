"""
Startup script for Healthcare Face Recognition System
Handles database initialization and server startup
"""
import os
import sys
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def initialize_database():
    """Initialize the database with proper tables"""
    try:
        # Ensure data directory exists
        data_dir = Path("data/encrypted_db")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database connection
        db_path = data_dir / "healthcare_fr.db"
        conn = sqlite3.connect(str(db_path))
        
        # Create patients table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                encrypted_data BLOB NOT NULL,
                facial_embedding_hash TEXT UNIQUE NOT NULL,
                consent_status BOOLEAN NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create identification_logs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS identification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                confidence_score REAL,
                processing_time_ms REAL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        """)
        
        # Create consent_records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consent_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                consent_type TEXT NOT NULL,
                status TEXT NOT NULL,
                granted_at TIMESTAMP,
                expires_at TIMESTAMP,
                withdrawn_at TIMESTAMP,
                consent_method TEXT DEFAULT 'digital',
                consent_version TEXT DEFAULT '1.0',
                ip_address TEXT,
                user_agent TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, consent_type)
            )
        """)
        
        # Create consent_audit_log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consent_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                action TEXT NOT NULL,
                consent_type TEXT,
                old_status TEXT,
                new_status TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                notes TEXT
            )
        """)
        
        # Create bias_measurements table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bias_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL NOT NULL,
                is_biased BOOLEAN NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """)
        
        # Create fairness_reports table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fairness_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                overall_fairness_score REAL NOT NULL,
                bias_measurements TEXT,
                recommendations TEXT,
                model_performance TEXT
            )
        """)
        
        # Create demographic_performance table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS demographic_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                demographic_group TEXT NOT NULL,
                attribute TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                false_positive_rate REAL NOT NULL,
                false_negative_rate REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        print("✓ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Main startup function"""
    print("🏥 Healthcare Face Recognition System - Starting...")
    print("=" * 50)
    
    # Initialize database
    if not initialize_database():
        print("Failed to initialize database. Exiting.")
        return 1
    
    # Start the main application
    print("Starting FastAPI server...")
    try:
        import uvicorn
        from config.settings import settings
        
        uvicorn.run(
            "main:app",
            host=settings.api_host,
            port=settings.api_port,
            workers=1,  # Use single worker for development
            reload=False,  # Disable reload for stability
            log_level=settings.system.log_level.lower()
        )
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
