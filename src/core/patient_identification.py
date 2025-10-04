"""
Privacy-First Patient Identification Module
Handles secure face recognition and patient identification with encryption
"""
import cv2
import numpy as np
import face_recognition
import hashlib
import base64
from typing import Optional, Dict, List, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import sqlite3
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from ..privacy.encryption import EncryptionManager
from ..privacy.consent_manager import ConsentManager
from ..security.anti_spoofing import AntiSpoofingDetector
from ..utils.face_utils import FaceUtils
from config.settings import settings


class IdentificationStatus(Enum):
    """Patient identification status"""
    IDENTIFIED = "identified"
    NOT_FOUND = "not_found"
    CONSENT_DENIED = "consent_denied"
    SPOOFING_DETECTED = "spoofing_detected"
    LOW_CONFIDENCE = "low_confidence"
    ERROR = "error"


@dataclass
class PatientInfo:
    """Patient information structure"""
    patient_id: str
    name: str
    date_of_birth: str
    medical_record_number: str
    consent_status: bool
    last_updated: datetime
    facial_embedding_hash: str


@dataclass
class IdentificationResult:
    """Result of patient identification"""
    status: IdentificationStatus
    patient_info: Optional[PatientInfo] = None
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None


class PatientIdentificationSystem:
    """
    Privacy-first patient identification system with encryption and consent management
    """
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.consent_manager = ConsentManager()
        self.anti_spoofing = AntiSpoofingDetector()
        self.face_utils = FaceUtils()
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Load face recognition model
        self._load_face_model()
    
    def _init_database(self):
        """Initialize encrypted patient database"""
        try:
            # Extract database path from URL
            db_url = settings.database.database_url
            if db_url.startswith("sqlite:///"):
                db_path = db_url[10:]  # Remove "sqlite:///" prefix
            else:
                db_path = db_url
            self.conn = sqlite3.connect(db_path)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    encrypted_data BLOB NOT NULL,
                    facial_embedding_hash TEXT UNIQUE NOT NULL,
                    consent_status BOOLEAN NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.execute("""
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
            
            self.conn.commit()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _load_face_model(self):
        """Load face recognition model"""
        try:
            # In a real implementation, you would load a pre-trained model
            # For now, we'll use the face_recognition library
            self.logger.info("Face recognition model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load face recognition model: {e}")
            raise
    
    def register_patient(self, 
                        patient_id: str,
                        name: str,
                        date_of_birth: str,
                        medical_record_number: str,
                        face_image: np.ndarray,
                        consent_given: bool = False) -> bool:
        """
        Register a new patient with encrypted facial data
        
        Args:
            patient_id: Unique patient identifier
            name: Patient's full name
            date_of_birth: Patient's date of birth
            medical_record_number: Medical record number
            face_image: Face image as numpy array
            consent_given: Whether patient has given consent
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Extract facial embedding
            face_encodings = face_recognition.face_encodings(face_image)
            if not face_encodings:
                self.logger.error(f"No face found in image for patient {patient_id}")
                return False
            
            face_encoding = face_encodings[0]
            
            # Create facial embedding hash for indexing
            embedding_hash = hashlib.sha256(face_encoding.tobytes()).hexdigest()
            
            # Create patient info
            patient_info = PatientInfo(
                patient_id=patient_id,
                name=name,
                date_of_birth=date_of_birth,
                medical_record_number=medical_record_number,
                consent_status=consent_given,
                last_updated=datetime.now(),
                facial_embedding_hash=embedding_hash
            )
            
            # Encrypt patient data
            encrypted_data = self.encryption_manager.encrypt_patient_data(patient_info)
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO patients 
                (patient_id, encrypted_data, facial_embedding_hash, consent_status)
                VALUES (?, ?, ?, ?)
            """, (patient_id, encrypted_data, embedding_hash, consent_given))
            
            self.conn.commit()
            
            # Log registration
            self.logger.info(f"Patient {patient_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register patient {patient_id}: {e}")
            return False
    
    def identify_patient(self, 
                        face_image: np.ndarray,
                        ip_address: str = None,
                        user_agent: str = None) -> IdentificationResult:
        """
        Identify patient from face image with privacy and security checks
        
        Args:
            face_image: Face image as numpy array
            ip_address: Client IP address for logging
            user_agent: Client user agent for logging
            
        Returns:
            IdentificationResult: Result of identification attempt
        """
        start_time = datetime.now()
        
        try:
            # Anti-spoofing check
            if not self.anti_spoofing.is_live_face(face_image):
                return IdentificationResult(
                    status=IdentificationStatus.SPOOFING_DETECTED,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    error_message="Spoofing detected"
                )
            
            # Extract facial embedding
            face_encodings = face_recognition.face_encodings(face_image)
            if not face_encodings:
                return IdentificationResult(
                    status=IdentificationStatus.ERROR,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    error_message="No face detected in image"
                )
            
            face_encoding = face_encodings[0]
            
            # Find matching patient
            match_result = self._find_matching_patient(face_encoding)
            
            if match_result is None:
                return IdentificationResult(
                    status=IdentificationStatus.NOT_FOUND,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            
            patient_id, confidence_score = match_result
            
            # Check consent
            if not self.consent_manager.has_consent(patient_id):
                return IdentificationResult(
                    status=IdentificationStatus.CONSENT_DENIED,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    error_message="Patient consent not given"
                )
            
            # Check confidence threshold
            if confidence_score < settings.model.confidence_threshold:
                return IdentificationResult(
                    status=IdentificationStatus.LOW_CONFIDENCE,
                    confidence_score=confidence_score,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    error_message=f"Confidence score {confidence_score} below threshold {settings.model.confidence_threshold}"
                )
            
            # Get patient info
            patient_info = self._get_patient_info(patient_id)
            
            # Log identification
            self._log_identification(patient_id, IdentificationStatus.IDENTIFIED, 
                                   confidence_score, start_time, ip_address, user_agent)
            
            return IdentificationResult(
                status=IdentificationStatus.IDENTIFIED,
                patient_info=patient_info,
                confidence_score=confidence_score,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Patient identification failed: {e}")
            return IdentificationResult(
                status=IdentificationStatus.ERROR,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )
    
    def _find_matching_patient(self, face_encoding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Find matching patient based on facial encoding"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT patient_id, encrypted_data FROM patients")
            
            best_match = None
            best_confidence = 0.0
            
            for row in cursor.fetchall():
                patient_id, encrypted_data = row
                
                # Decrypt patient data
                patient_info = self.encryption_manager.decrypt_patient_data(encrypted_data)
                
                # Reconstruct face encoding from stored data
                # In a real implementation, you would store the actual encoding
                # For now, we'll use a simplified approach
                stored_encoding = self._get_stored_encoding(patient_id)
                
                if stored_encoding is not None:
                    # Calculate face distance
                    face_distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
                    confidence = 1 - face_distance
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = patient_id
            
            if best_match and best_confidence > settings.model.confidence_threshold:
                return best_match, best_confidence
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding matching patient: {e}")
            return None
    
    def _get_stored_encoding(self, patient_id: str) -> Optional[np.ndarray]:
        """Get stored face encoding for patient"""
        # In a real implementation, you would retrieve the actual stored encoding
        # For now, this is a placeholder
        return None
    
    def _get_patient_info(self, patient_id: str) -> Optional[PatientInfo]:
        """Get patient information by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT encrypted_data FROM patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()
            
            if row:
                encrypted_data = row[0]
                return self.encryption_manager.decrypt_patient_data(encrypted_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting patient info for {patient_id}: {e}")
            return None
    
    def _log_identification(self, patient_id: str, status: IdentificationStatus, 
                          confidence_score: float, start_time: datetime,
                          ip_address: str = None, user_agent: str = None):
        """Log identification attempt"""
        try:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO identification_logs 
                (patient_id, status, confidence_score, processing_time_ms, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (patient_id, status.value, confidence_score, processing_time, ip_address, user_agent))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to log identification: {e}")
    
    def update_consent(self, patient_id: str, consent_given: bool) -> bool:
        """Update patient consent status"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE patients 
                SET consent_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE patient_id = ?
            """, (consent_given, patient_id))
            
            self.conn.commit()
            
            # Update consent manager
            if consent_given:
                self.consent_manager.grant_consent(patient_id)
            else:
                self.consent_manager.revoke_consent(patient_id)
            
            self.logger.info(f"Consent updated for patient {patient_id}: {consent_given}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update consent for patient {patient_id}: {e}")
            return False
    
    def get_identification_stats(self) -> Dict:
        """Get identification statistics"""
        try:
            cursor = self.conn.cursor()
            
            # Get total patients
            cursor.execute("SELECT COUNT(*) FROM patients")
            total_patients = cursor.fetchone()[0]
            
            # Get patients with consent
            cursor.execute("SELECT COUNT(*) FROM patients WHERE consent_status = 1")
            patients_with_consent = cursor.fetchone()[0]
            
            # Get recent identification attempts
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM identification_logs 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY status
            """)
            recent_attempts = dict(cursor.fetchall())
            
            return {
                "total_patients": total_patients,
                "patients_with_consent": patients_with_consent,
                "recent_identification_attempts": recent_attempts,
                "consent_rate": patients_with_consent / total_patients if total_patients > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get identification stats: {e}")
            return {}
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        try:
            retention_date = datetime.now() - timedelta(days=settings.privacy.data_retention_days)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM identification_logs 
                WHERE timestamp < ?
            """, (retention_date,))
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} old identification logs")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
