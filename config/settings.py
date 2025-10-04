"""
Configuration settings for the Healthcare Face Recognition System
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class SecuritySettings(BaseSettings):
    """Security-related configuration"""
    encryption_key: str = Field(default="", description="Base64 encoded encryption key")
    jwt_secret: str = Field(default="", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")
    max_login_attempts: int = Field(default=3, description="Maximum login attempts before lockout")
    session_timeout_minutes: int = Field(default=30, description="Session timeout in minutes")
    
    class Config:
        env_prefix = "SECURITY_"


class PrivacySettings(BaseSettings):
    """Privacy and compliance settings"""
    enable_consent_management: bool = Field(default=True, description="Enable patient consent management")
    data_retention_days: int = Field(default=365, description="Data retention period in days")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    anonymize_logs: bool = Field(default=True, description="Anonymize audit logs")
    gdpr_compliance: bool = Field(default=True, description="Enable GDPR compliance features")
    hipaa_compliance: bool = Field(default=True, description="Enable HIPAA compliance features")
    
    class Config:
        env_prefix = "PRIVACY_"


class ModelSettings(BaseSettings):
    """AI model configuration"""
    face_recognition_model: str = Field(default="facenet", description="Face recognition model to use")
    genetic_screening_model: str = Field(default="genetic_classifier", description="Genetic screening model")
    liveness_detection_model: str = Field(default="liveness_net", description="Liveness detection model")
    model_update_frequency_days: int = Field(default=30, description="Model update frequency in days")
    confidence_threshold: float = Field(default=0.85, description="Minimum confidence threshold for face recognition")
    genetic_screening_threshold: float = Field(default=0.7, description="Threshold for genetic screening alerts")
    
    class Config:
        env_prefix = "MODEL_"


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    database_url: str = Field(default="sqlite:///./data/encrypted_db/healthcare_fr.db", description="Database URL")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL for caching")
    backup_frequency_hours: int = Field(default=24, description="Database backup frequency in hours")
    max_connections: int = Field(default=100, description="Maximum database connections")
    
    class Config:
        env_prefix = "DATABASE_"


class BiasDetectionSettings(BaseSettings):
    """Bias detection and fairness settings"""
    enable_bias_monitoring: bool = Field(default=True, description="Enable bias monitoring")
    fairness_metrics: List[str] = Field(default=["demographic_parity", "equalized_odds"], description="Fairness metrics to track")
    bias_threshold: float = Field(default=0.1, description="Bias threshold for alerts")
    demographic_attributes: List[str] = Field(default=["age", "gender", "ethnicity"], description="Demographic attributes to monitor")
    
    class Config:
        env_prefix = "BIAS_"


class SystemSettings(BaseSettings):
    """General system settings"""
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    supported_image_formats: List[str] = Field(default=["jpg", "jpeg", "png", "bmp"], description="Supported image formats")
    camera_resolution: tuple = Field(default=(1920, 1080), description="Camera resolution")
    processing_fps: int = Field(default=30, description="Processing FPS")
    
    class Config:
        env_prefix = "SYSTEM_"


class Settings(BaseSettings):
    """Main settings class combining all configuration"""
    security: SecuritySettings = SecuritySettings()
    privacy: PrivacySettings = PrivacySettings()
    model: ModelSettings = ModelSettings()
    database: DatabaseSettings = DatabaseSettings()
    bias_detection: BiasDetectionSettings = BiasDetectionSettings()
    system: SystemSettings = SystemSettings()
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8001, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    
    # File paths
    data_dir: str = Field(default="./data", description="Data directory")
    models_dir: str = Field(default="./data/models", description="Models directory")
    logs_dir: str = Field(default="./data/logs", description="Logs directory")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
