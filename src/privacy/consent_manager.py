"""
Consent Management System for Healthcare Face Recognition
Handles patient consent for data processing and storage
"""
import json
import sqlite3
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import logging

from config.settings import settings


class ConsentType(Enum):
    """Types of consent"""
    FACE_RECOGNITION = "face_recognition"
    DATA_STORAGE = "data_storage"
    GENETIC_SCREENING = "genetic_screening"
    RESEARCH_USE = "research_use"
    THIRD_PARTY_SHARING = "third_party_sharing"
    AUDIT_LOGGING = "audit_logging"


class ConsentStatus(Enum):
    """Consent status"""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


@dataclass
class ConsentRecord:
    """Consent record structure"""
    patient_id: str
    consent_type: ConsentType
    status: ConsentStatus
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    consent_method: str = "digital"  # digital, paper, verbal
    consent_version: str = "1.0"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    notes: Optional[str] = None


class ConsentManager:
    """
    Manages patient consent for data processing and storage
    Implements GDPR and HIPAA compliant consent management
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conn = self._init_database()
        self._consent_cache: Dict[str, Set[ConsentType]] = {}
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize consent database"""
        try:
            # Extract database path from URL
            db_url = settings.database.database_url
            if db_url.startswith("sqlite:///"):
                db_path = db_url[10:]  # Remove "sqlite:///" prefix
            else:
                db_path = db_url
            conn = sqlite3.connect(db_path)
            
            # Create consent table
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
            
            # Create consent audit log
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
            
            conn.commit()
            self.logger.info("Consent database initialized successfully")
            return conn
            
        except Exception as e:
            self.logger.error(f"Failed to initialize consent database: {e}")
            raise
    
    def grant_consent(self, 
                     patient_id: str,
                     consent_type: ConsentType,
                     expires_in_days: Optional[int] = None,
                     consent_method: str = "digital",
                     ip_address: Optional[str] = None,
                     user_agent: Optional[str] = None,
                     notes: Optional[str] = None) -> bool:
        """
        Grant consent for a specific type of data processing
        
        Args:
            patient_id: Patient identifier
            consent_type: Type of consent
            expires_in_days: Consent expiration in days (None for no expiration)
            consent_method: Method of consent (digital, paper, verbal)
            ip_address: IP address of consent giver
            user_agent: User agent of consent giver
            notes: Additional notes
            
        Returns:
            bool: True if consent granted successfully
        """
        try:
            granted_at = datetime.now()
            expires_at = None
            
            if expires_in_days:
                expires_at = granted_at + timedelta(days=expires_in_days)
            
            # Check if consent already exists
            existing_consent = self.get_consent_status(patient_id, consent_type)
            
            cursor = self.conn.cursor()
            
            if existing_consent:
                # Update existing consent
                cursor.execute("""
                    UPDATE consent_records 
                    SET status = ?, granted_at = ?, expires_at = ?, withdrawn_at = NULL,
                        consent_method = ?, ip_address = ?, user_agent = ?, notes = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE patient_id = ? AND consent_type = ?
                """, (ConsentStatus.GRANTED.value, granted_at, expires_at, consent_method,
                      ip_address, user_agent, notes, patient_id, consent_type.value))
            else:
                # Insert new consent
                cursor.execute("""
                    INSERT INTO consent_records 
                    (patient_id, consent_type, status, granted_at, expires_at, 
                     consent_method, ip_address, user_agent, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (patient_id, consent_type.value, ConsentStatus.GRANTED.value, 
                      granted_at, expires_at, consent_method, ip_address, user_agent, notes))
            
            # Log consent action
            self._log_consent_action(patient_id, "GRANT", consent_type, 
                                   existing_consent, ConsentStatus.GRANTED,
                                   ip_address, user_agent, notes)
            
            self.conn.commit()
            
            # Update cache
            self._update_consent_cache(patient_id, consent_type, ConsentStatus.GRANTED)
            
            self.logger.info(f"Consent granted for patient {patient_id}, type {consent_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to grant consent for patient {patient_id}: {e}")
            return False
    
    def revoke_consent(self, 
                      patient_id: str,
                      consent_type: ConsentType,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      notes: Optional[str] = None) -> bool:
        """
        Revoke consent for a specific type of data processing
        
        Args:
            patient_id: Patient identifier
            consent_type: Type of consent to revoke
            ip_address: IP address of person revoking consent
            user_agent: User agent of person revoking consent
            notes: Additional notes
            
        Returns:
            bool: True if consent revoked successfully
        """
        try:
            withdrawn_at = datetime.now()
            old_status = self.get_consent_status(patient_id, consent_type)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE consent_records 
                SET status = ?, withdrawn_at = ?, updated_at = CURRENT_TIMESTAMP
                WHERE patient_id = ? AND consent_type = ?
            """, (ConsentStatus.WITHDRAWN.value, withdrawn_at, patient_id, consent_type.value))
            
            # Log consent action
            self._log_consent_action(patient_id, "REVOKE", consent_type, 
                                   old_status, ConsentStatus.WITHDRAWN,
                                   ip_address, user_agent, notes)
            
            self.conn.commit()
            
            # Update cache
            self._update_consent_cache(patient_id, consent_type, ConsentStatus.WITHDRAWN)
            
            self.logger.info(f"Consent revoked for patient {patient_id}, type {consent_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to revoke consent for patient {patient_id}: {e}")
            return False
    
    def has_consent(self, patient_id: str, consent_type: ConsentType) -> bool:
        """
        Check if patient has valid consent for a specific type
        
        Args:
            patient_id: Patient identifier
            consent_type: Type of consent to check
            
        Returns:
            bool: True if patient has valid consent
        """
        try:
            # Check cache first
            if patient_id in self._consent_cache:
                if consent_type in self._consent_cache[patient_id]:
                    return True
            
            # Check database
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT status, expires_at 
                FROM consent_records 
                WHERE patient_id = ? AND consent_type = ?
            """, (patient_id, consent_type.value))
            
            row = cursor.fetchone()
            if not row:
                return False
            
            status, expires_at = row
            
            # Check if consent is granted and not expired
            if status == ConsentStatus.GRANTED.value:
                if expires_at is None:
                    # No expiration
                    self._update_consent_cache(patient_id, consent_type, ConsentStatus.GRANTED)
                    return True
                else:
                    # Check expiration
                    expires_datetime = datetime.fromisoformat(expires_at)
                    if datetime.now() < expires_datetime:
                        self._update_consent_cache(patient_id, consent_type, ConsentStatus.GRANTED)
                        return True
                    else:
                        # Expired
                        self._update_consent_cache(patient_id, consent_type, ConsentStatus.EXPIRED)
                        return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check consent for patient {patient_id}: {e}")
            return False
    
    def get_consent_status(self, patient_id: str, consent_type: ConsentType) -> Optional[ConsentStatus]:
        """
        Get current consent status for a patient and consent type
        
        Args:
            patient_id: Patient identifier
            consent_type: Type of consent
            
        Returns:
            Optional[ConsentStatus]: Current consent status
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT status, expires_at 
                FROM consent_records 
                WHERE patient_id = ? AND consent_type = ?
            """, (patient_id, consent_type.value))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            status, expires_at = row
            
            # Check expiration
            if status == ConsentStatus.GRANTED.value and expires_at:
                expires_datetime = datetime.fromisoformat(expires_at)
                if datetime.now() >= expires_datetime:
                    return ConsentStatus.EXPIRED
            
            return ConsentStatus(status)
            
        except Exception as e:
            self.logger.error(f"Failed to get consent status for patient {patient_id}: {e}")
            return None
    
    def get_all_consents(self, patient_id: str) -> Dict[ConsentType, ConsentStatus]:
        """
        Get all consent statuses for a patient
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict: Mapping of consent types to their statuses
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT consent_type, status, expires_at 
                FROM consent_records 
                WHERE patient_id = ?
            """, (patient_id,))
            
            consents = {}
            for row in cursor.fetchall():
                consent_type, status, expires_at = row
                
                # Check expiration
                if status == ConsentStatus.GRANTED.value and expires_at:
                    expires_datetime = datetime.fromisoformat(expires_at)
                    if datetime.now() >= expires_datetime:
                        status = ConsentStatus.EXPIRED.value
                
                consents[ConsentType(consent_type)] = ConsentStatus(status)
            
            return consents
            
        except Exception as e:
            self.logger.error(f"Failed to get all consents for patient {patient_id}: {e}")
            return {}
    
    def get_consent_summary(self, patient_id: str) -> Dict:
        """
        Get a summary of all consents for a patient
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict: Consent summary
        """
        try:
            consents = self.get_all_consents(patient_id)
            
            summary = {
                "patient_id": patient_id,
                "total_consent_types": len(ConsentType),
                "granted_consents": [],
                "denied_consents": [],
                "expired_consents": [],
                "withdrawn_consents": [],
                "pending_consents": []
            }
            
            for consent_type, status in consents.items():
                if status == ConsentStatus.GRANTED:
                    summary["granted_consents"].append(consent_type.value)
                elif status == ConsentStatus.DENIED:
                    summary["denied_consents"].append(consent_type.value)
                elif status == ConsentStatus.EXPIRED:
                    summary["expired_consents"].append(consent_type.value)
                elif status == ConsentStatus.WITHDRAWN:
                    summary["withdrawn_consents"].append(consent_type.value)
                elif status == ConsentStatus.PENDING:
                    summary["pending_consents"].append(consent_type.value)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get consent summary for patient {patient_id}: {e}")
            return {}
    
    def cleanup_expired_consents(self) -> int:
        """
        Clean up expired consents and update their status
        
        Returns:
            int: Number of consents updated
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE consent_records 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE status = ? AND expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
            """, (ConsentStatus.EXPIRED.value, ConsentStatus.GRANTED.value))
            
            updated_count = cursor.rowcount
            self.conn.commit()
            
            if updated_count > 0:
                self.logger.info(f"Updated {updated_count} expired consents")
            
            return updated_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired consents: {e}")
            return 0
    
    def _log_consent_action(self, 
                           patient_id: str,
                           action: str,
                           consent_type: ConsentType,
                           old_status: Optional[ConsentStatus],
                           new_status: ConsentStatus,
                           ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None,
                           notes: Optional[str] = None):
        """Log consent action for audit trail"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO consent_audit_log 
                (patient_id, action, consent_type, old_status, new_status, 
                 ip_address, user_agent, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, action, consent_type.value, 
                  old_status.value if old_status else None,
                  new_status.value, ip_address, user_agent, notes))
            
        except Exception as e:
            self.logger.error(f"Failed to log consent action: {e}")
    
    def _update_consent_cache(self, 
                             patient_id: str,
                             consent_type: ConsentType,
                             status: ConsentStatus):
        """Update consent cache"""
        try:
            if patient_id not in self._consent_cache:
                self._consent_cache[patient_id] = set()
            
            if status == ConsentStatus.GRANTED:
                self._consent_cache[patient_id].add(consent_type)
            else:
                self._consent_cache[patient_id].discard(consent_type)
                
        except Exception as e:
            self.logger.error(f"Failed to update consent cache: {e}")
    
    def get_consent_statistics(self) -> Dict:
        """Get consent statistics"""
        try:
            cursor = self.conn.cursor()
            
            # Get total consents by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM consent_records 
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Get consents by type
            cursor.execute("""
                SELECT consent_type, COUNT(*) 
                FROM consent_records 
                GROUP BY consent_type
            """)
            type_counts = dict(cursor.fetchall())
            
            # Get recent consent activities
            cursor.execute("""
                SELECT action, COUNT(*) 
                FROM consent_audit_log 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY action
            """)
            recent_activities = dict(cursor.fetchall())
            
            return {
                "status_counts": status_counts,
                "type_counts": type_counts,
                "recent_activities": recent_activities,
                "total_patients_with_consent": len(self._consent_cache)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get consent statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
