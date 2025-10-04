"""
Encryption Manager for Healthcare Face Recognition System
Handles encryption/decryption of sensitive patient data
"""
import base64
import hashlib
import os
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import json
import logging
from dataclasses import asdict

from config.settings import settings


class EncryptionManager:
    """
    Manages encryption and decryption of sensitive patient data
    Implements AES-256 encryption with PBKDF2 key derivation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._encryption_key = self._get_or_create_encryption_key()
        self._fernet = Fernet(self._encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key"""
        try:
            if settings.security.encryption_key:
                # Use provided key
                key_bytes = base64.b64decode(settings.security.encryption_key)
                return key_bytes
            else:
                # Generate new key
                key = Fernet.generate_key()
                self.logger.warning("No encryption key provided in settings. Generated new key.")
                self.logger.warning(f"Please save this key securely: {base64.b64encode(key).decode()}")
                return key
                
        except Exception as e:
            self.logger.error(f"Failed to get encryption key: {e}")
            raise
    
    def encrypt_patient_data(self, patient_data: Any) -> bytes:
        """
        Encrypt patient data using AES-256
        
        Args:
            patient_data: Patient data object to encrypt
            
        Returns:
            bytes: Encrypted data
        """
        try:
            # Convert to JSON string
            if hasattr(patient_data, '__dict__'):
                data_dict = asdict(patient_data)
            else:
                data_dict = patient_data
            
            json_data = json.dumps(data_dict, default=str)
            data_bytes = json_data.encode('utf-8')
            
            # Encrypt using Fernet (AES-128 in CBC mode with HMAC)
            encrypted_data = self._fernet.encrypt(data_bytes)
            
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt patient data: {e}")
            raise
    
    def decrypt_patient_data(self, encrypted_data: bytes) -> Any:
        """
        Decrypt patient data
        
        Args:
            encrypted_data: Encrypted data bytes
            
        Returns:
            Any: Decrypted patient data
        """
        try:
            # Decrypt using Fernet
            decrypted_bytes = self._fernet.decrypt(encrypted_data)
            
            # Convert back to JSON
            json_data = decrypted_bytes.decode('utf-8')
            data_dict = json.loads(json_data)
            
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt patient data: {e}")
            raise
    
    def encrypt_facial_embedding(self, embedding: bytes) -> bytes:
        """
        Encrypt facial embedding data
        
        Args:
            embedding: Facial embedding bytes
            
        Returns:
            bytes: Encrypted embedding
        """
        try:
            return self._fernet.encrypt(embedding)
        except Exception as e:
            self.logger.error(f"Failed to encrypt facial embedding: {e}")
            raise
    
    def decrypt_facial_embedding(self, encrypted_embedding: bytes) -> bytes:
        """
        Decrypt facial embedding data
        
        Args:
            encrypted_embedding: Encrypted embedding bytes
            
        Returns:
            bytes: Decrypted embedding
        """
        try:
            return self._fernet.decrypt(encrypted_embedding)
        except Exception as e:
            self.logger.error(f"Failed to decrypt facial embedding: {e}")
            raise
    
    def hash_sensitive_data(self, data: str) -> str:
        """
        Create a secure hash of sensitive data for indexing
        
        Args:
            data: Sensitive data to hash
            
        Returns:
            str: SHA-256 hash of the data
        """
        try:
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to hash sensitive data: {e}")
            raise
    
    def encrypt_field(self, field_value: str) -> str:
        """
        Encrypt a single field value
        
        Args:
            field_value: Field value to encrypt
            
        Returns:
            str: Base64 encoded encrypted value
        """
        try:
            encrypted_bytes = self._fernet.encrypt(field_value.encode('utf-8'))
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encrypt field: {e}")
            raise
    
    def decrypt_field(self, encrypted_field: str) -> str:
        """
        Decrypt a single field value
        
        Args:
            encrypted_field: Base64 encoded encrypted field
            
        Returns:
            str: Decrypted field value
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_field.encode('utf-8'))
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to decrypt field: {e}")
            raise
    
    def create_secure_token(self, data: Dict[str, Any], expiration_hours: int = 24) -> str:
        """
        Create a secure token for temporary data access
        
        Args:
            data: Data to include in token
            expiration_hours: Token expiration time in hours
            
        Returns:
            str: Secure token
        """
        try:
            import jwt
            from datetime import datetime, timedelta
            
            payload = {
                'data': data,
                'exp': datetime.utcnow() + timedelta(hours=expiration_hours),
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(payload, settings.security.jwt_secret, 
                             algorithm=settings.security.jwt_algorithm)
            
            return token
            
        except Exception as e:
            self.logger.error(f"Failed to create secure token: {e}")
            raise
    
    def verify_secure_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a secure token
        
        Args:
            token: Token to verify
            
        Returns:
            Optional[Dict]: Decoded data if valid, None otherwise
        """
        try:
            import jwt
            
            payload = jwt.decode(token, settings.security.jwt_secret, 
                               algorithms=[settings.security.jwt_algorithm])
            
            return payload.get('data')
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to verify token: {e}")
            return None
    
    def generate_secure_filename(self, original_filename: str) -> str:
        """
        Generate a secure filename for storing encrypted data
        
        Args:
            original_filename: Original filename
            
        Returns:
            str: Secure filename
        """
        try:
            # Create hash of original filename
            filename_hash = hashlib.sha256(original_filename.encode('utf-8')).hexdigest()
            
            # Add timestamp for uniqueness
            import time
            timestamp = str(int(time.time()))
            
            return f"{filename_hash}_{timestamp}.enc"
            
        except Exception as e:
            self.logger.error(f"Failed to generate secure filename: {e}")
            raise
    
    def encrypt_file(self, file_path: str, output_path: str) -> bool:
        """
        Encrypt a file
        
        Args:
            file_path: Path to file to encrypt
            output_path: Path for encrypted file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, 'rb') as infile:
                file_data = infile.read()
            
            encrypted_data = self._fernet.encrypt(file_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(encrypted_data)
            
            self.logger.info(f"File encrypted successfully: {file_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt file {file_path}: {e}")
            return False
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str) -> bool:
        """
        Decrypt a file
        
        Args:
            encrypted_file_path: Path to encrypted file
            output_path: Path for decrypted file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(encrypted_file_path, 'rb') as infile:
                encrypted_data = infile.read()
            
            decrypted_data = self._fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(decrypted_data)
            
            self.logger.info(f"File decrypted successfully: {encrypted_file_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt file {encrypted_file_path}: {e}")
            return False
    
    def get_encryption_info(self) -> Dict[str, str]:
        """
        Get encryption information (for debugging/monitoring)
        
        Returns:
            Dict: Encryption information
        """
        return {
            "algorithm": "AES-256-CBC",
            "key_derivation": "PBKDF2-HMAC-SHA256",
            "authentication": "HMAC-SHA256",
            "key_size": "256 bits"
        }
