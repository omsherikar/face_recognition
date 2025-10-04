"""
Healthcare Face Recognition System - Main Application
Secure, privacy-preserving face recognition for healthcare environments
"""
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import cv2
import numpy as np 
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.patient_identification import PatientIdentificationSystem, IdentificationResult, IdentificationStatus
from src.genetic_screening.genetic_analyzer import GeneticAnalyzer, GeneticScreeningResult
from src.bias_detection.bias_monitor import BiasMonitor, FairnessReport
from src.privacy.consent_manager import ConsentManager, ConsentType, ConsentStatus
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.system.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{settings.logs_dir}/healthcare_fr.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Face Recognition System",
    description="Secure, privacy-preserving face recognition for healthcare environments",
    version="1.0.0",
    docs_url="/docs" if settings.system.debug else None,
    redoc_url="/redoc" if settings.system.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.system.debug else ["https://your-hospital-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global system instances
patient_system: Optional[PatientIdentificationSystem] = None
genetic_analyzer: Optional[GeneticAnalyzer] = None
bias_monitor: Optional[BiasMonitor] = None
consent_manager: Optional[ConsentManager] = None


# Pydantic models
class PatientRegistrationRequest(BaseModel):
    patient_id: str
    name: str
    date_of_birth: str
    medical_record_number: str
    consent_given: bool = False


class PatientIdentificationRequest(BaseModel):
    session_id: str = "default"


class ConsentUpdateRequest(BaseModel):
    patient_id: str
    consent_type: str
    consent_given: bool


class GeneticScreeningRequest(BaseModel):
    session_id: str = "default"
    enable_screening: bool = True


class IdentificationResponse(BaseModel):
    status: str
    patient_id: Optional[str] = None
    name: Optional[str] = None
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    genetic_screening_results: Optional[list] = None
    error_message: Optional[str] = None


class SystemStatusResponse(BaseModel):
    status: str
    components: dict
    statistics: dict


# Dependency to get current user (placeholder for authentication)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In a real implementation, you would validate the JWT token
    # For now, we'll just return a placeholder user
    return {"user_id": "admin", "role": "healthcare_provider"}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global patient_system, genetic_analyzer, bias_monitor, consent_manager
    
    try:
        logger.info("Starting Healthcare Face Recognition System...")
        
        # Create necessary directories
        os.makedirs(settings.data_dir, exist_ok=True)
        os.makedirs(settings.models_dir, exist_ok=True)
        os.makedirs(settings.logs_dir, exist_ok=True)
        
        # Initialize system components
        patient_system = PatientIdentificationSystem()
        genetic_analyzer = GeneticAnalyzer()
        bias_monitor = BiasMonitor()
        consent_manager = ConsentManager()
        
        logger.info("System components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global patient_system, genetic_analyzer, bias_monitor, consent_manager
    
    try:
        logger.info("Shutting down Healthcare Face Recognition System...")
        
        if patient_system:
            patient_system.close()
        if bias_monitor:
            bias_monitor.close()
        if consent_manager:
            consent_manager.close()
        
        logger.info("System shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# API Endpoints

@app.get("/", response_model=SystemStatusResponse)
async def root():
    """Get system status"""
    try:
        components = {
            "patient_identification": patient_system is not None,
            "genetic_screening": genetic_analyzer is not None,
            "bias_monitoring": bias_monitor is not None,
            "consent_management": consent_manager is not None
        }
        
        statistics = {}
        if patient_system:
            statistics["identification"] = patient_system.get_identification_stats()
        if bias_monitor:
            statistics["bias_detection"] = bias_monitor.get_bias_statistics()
        if consent_manager:
            statistics["consent"] = consent_manager.get_consent_statistics()
        
        return SystemStatusResponse(
            status="operational",
            components=components,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register-patient")
async def register_patient(
    patient_id: str = Form(...),
    name: str = Form(...),
    date_of_birth: str = Form(...),
    medical_record_number: str = Form(...),
    consent_given: bool = Form(False),
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Register a new patient with facial data"""
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Register patient
        success = patient_system.register_patient(
            patient_id=patient_id,
            name=name,
            date_of_birth=date_of_birth,
            medical_record_number=medical_record_number,
            face_image=face_image,
            consent_given=consent_given
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to register patient")
        
        # Update consent if provided
        if consent_given:
            consent_manager.grant_consent(
                patient_id=patient_id,
                consent_type=ConsentType.FACE_RECOGNITION
            )
        
        return {"message": "Patient registered successfully", "patient_id": patient_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify-patient", response_model=IdentificationResponse)
async def identify_patient(
    image: UploadFile = File(...),
    request_data: PatientIdentificationRequest = Depends(),
    current_user: dict = Depends(get_current_user)
):
    """Identify patient from facial image"""
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Identify patient
        result: IdentificationResult = patient_system.identify_patient(
            face_image=face_image,
            ip_address="127.0.0.1",  # In real implementation, get from request
            user_agent="Healthcare-FR-API"
        )
        
        # Prepare response
        response = IdentificationResponse(
            status=result.status.value,
            confidence_score=result.confidence_score,
            processing_time_ms=result.processing_time_ms,
            error_message=result.error_message
        )
        
        if result.status == IdentificationStatus.IDENTIFIED and result.patient_info:
            response.patient_id = result.patient_info.patient_id
            response.name = result.patient_info.name
            
            # Optional genetic screening
            if genetic_analyzer and consent_manager.has_consent(
                result.patient_info.patient_id, ConsentType.GENETIC_SCREENING
            ):
                screening_results = genetic_analyzer.screen_for_genetic_disorders(face_image)
                response.genetic_screening_results = [
                    {
                        "disorder": result.disorder.value,
                        "risk_score": result.risk_score,
                        "confidence": result.confidence,
                        "recommendation": result.recommendation,
                        "disclaimer": result.disclaimer
                    }
                    for result in screening_results
                ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to identify patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-consent")
async def update_consent(
    consent_data: ConsentUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update patient consent"""
    try:
        consent_type = ConsentType(consent_data.consent_type)
        
        if consent_data.consent_given:
            success = consent_manager.grant_consent(
                patient_id=consent_data.patient_id,
                consent_type=consent_type
            )
        else:
            success = consent_manager.revoke_consent(
                patient_id=consent_data.patient_id,
                consent_type=consent_type
            )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update consent")
        
        return {"message": "Consent updated successfully"}
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid consent type")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/consent-status/{patient_id}")
async def get_consent_status(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get patient consent status"""
    try:
        consent_summary = consent_manager.get_consent_summary(patient_id)
        return consent_summary
        
    except Exception as e:
        logger.error(f"Failed to get consent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/genetic-screening/{patient_id}")
async def get_genetic_screening_results(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get genetic screening results for a patient"""
    try:
        # Check consent
        if not consent_manager.has_consent(patient_id, ConsentType.GENETIC_SCREENING):
            raise HTTPException(status_code=403, detail="Patient has not consented to genetic screening")
        
        # In a real implementation, you would retrieve stored results
        # For now, return a placeholder
        return {
            "patient_id": patient_id,
            "screening_results": [],
            "message": "No screening results available"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get genetic screening results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fairness-report")
async def get_fairness_report(
    current_user: dict = Depends(get_current_user)
):
    """Get fairness and bias monitoring report"""
    try:
        report = bias_monitor.generate_fairness_report()
        
        if not report:
            raise HTTPException(status_code=500, detail="Failed to generate fairness report")
        
        return {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "overall_fairness_score": report.overall_fairness_score,
            "bias_measurements": [
                {
                    "metric": measurement.metric.value,
                    "attribute": measurement.attribute.value,
                    "value": measurement.value,
                    "is_biased": measurement.is_biased,
                    "confidence": measurement.confidence
                }
                for measurement in report.bias_measurements
            ],
            "recommendations": report.recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fairness report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system-statistics")
async def get_system_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive system statistics"""
    try:
        statistics = {}
        
        if patient_system:
            statistics["identification"] = patient_system.get_identification_stats()
        
        if bias_monitor:
            statistics["bias_detection"] = bias_monitor.get_bias_statistics()
        
        if consent_manager:
            statistics["consent"] = consent_manager.get_consent_statistics()
        
        if genetic_analyzer:
            statistics["genetic_screening"] = genetic_analyzer.get_screening_statistics()
        
        return statistics
        
    except Exception as e:
        logger.error(f"Failed to get system statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup-data")
async def cleanup_old_data(
    current_user: dict = Depends(get_current_user)
):
    """Clean up old data based on retention policy"""
    try:
        if patient_system:
            patient_system.cleanup_old_data()
        
        if consent_manager:
            consent_manager.cleanup_expired_consents()
        
        return {"message": "Data cleanup completed successfully"}
        
    except Exception as e:
        logger.error(f"Failed to cleanup data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.system.debug,
        log_level=settings.system.log_level.lower()
    )
