"""
Genetic Disorder Pre-Screening Module for Healthcare Face Recognition System
AI-based facial feature analysis for genetic disorder screening
"""
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from ..utils.face_utils import FaceUtils
from config.settings import settings


class GeneticDisorder(Enum):
    """Supported genetic disorders for screening"""
    DOWN_SYNDROME = "down_syndrome"
    WILLIAMS_SYNDROME = "williams_syndrome"
    ANGELMAN_SYNDROME = "angelman_syndrome"
    PRADER_WILLI_SYNDROME = "prader_willi_syndrome"
    NOONAN_SYNDROME = "noonan_syndrome"
    TURNER_SYNDROME = "turner_syndrome"
    MARFAN_SYNDROME = "marfan_syndrome"
    ACHONDROPLASIA = "achondroplasia"


@dataclass
class FacialFeature:
    """Facial feature measurement"""
    name: str
    value: float
    confidence: float
    description: str


@dataclass
class GeneticScreeningResult:
    """Result of genetic disorder screening"""
    disorder: GeneticDisorder
    risk_score: float
    confidence: float
    facial_features: List[FacialFeature]
    recommendation: str
    processing_time_ms: float
    disclaimer: str


class GeneticAnalyzer:
    """
    AI-based genetic disorder screening through facial feature analysis
    Provides preliminary risk assessment for genetic disorders
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_utils = FaceUtils()
        
        # Load genetic screening models
        self.models = self._load_genetic_models()
        
        # Facial feature definitions
        self.feature_definitions = self._load_feature_definitions()
        
        # Risk thresholds
        self.risk_thresholds = {
            GeneticDisorder.DOWN_SYNDROME: 0.7,
            GeneticDisorder.WILLIAMS_SYNDROME: 0.6,
            GeneticDisorder.ANGELMAN_SYNDROME: 0.65,
            GeneticDisorder.PRADER_WILLI_SYNDROME: 0.6,
            GeneticDisorder.NOONAN_SYNDROME: 0.65,
            GeneticDisorder.TURNER_SYNDROME: 0.7,
            GeneticDisorder.MARFAN_SYNDROME: 0.6,
            GeneticDisorder.ACHONDROPLASIA: 0.65
        }
    
    def _load_genetic_models(self) -> Dict[GeneticDisorder, tf.keras.Model]:
        """Load pre-trained genetic screening models"""
        try:
            models = {}
            
            # In a real implementation, you would load actual trained models
            # For now, we'll create placeholder models
            for disorder in GeneticDisorder:
                # Create a simple placeholder model
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu', input_shape=(68,)),  # 68 facial landmarks
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                
                # Compile model
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                models[disorder] = model
            
            self.logger.info("Genetic screening models loaded (placeholder)")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to load genetic models: {e}")
            return {}
    
    def _load_feature_definitions(self) -> Dict[str, Dict]:
        """Load facial feature definitions for genetic disorders"""
        return {
            "down_syndrome": {
                "features": [
                    {"name": "epicanthal_fold", "description": "Presence of epicanthal fold"},
                    {"name": "flat_nasal_bridge", "description": "Flattened nasal bridge"},
                    {"name": "small_ears", "description": "Small, low-set ears"},
                    {"name": "protruding_tongue", "description": "Protruding tongue"},
                    {"name": "short_neck", "description": "Short neck length"},
                    {"name": "wide_hand_gap", "description": "Wide gap between first and second toes"}
                ]
            },
            "williams_syndrome": {
                "features": [
                    {"name": "wide_mouth", "description": "Wide mouth with full lips"},
                    {"name": "small_upturned_nose", "description": "Small, upturned nose"},
                    {"name": "full_cheeks", "description": "Full, rounded cheeks"},
                    {"name": "wide_set_eyes", "description": "Wide-set eyes"},
                    {"name": "small_chin", "description": "Small, pointed chin"}
                ]
            },
            "angelman_syndrome": {
                "features": [
                    {"name": "wide_mouth", "description": "Wide mouth with frequent smiling"},
                    {"name": "protruding_tongue", "description": "Protruding tongue"},
                    {"name": "wide_set_teeth", "description": "Wide-set teeth"},
                    {"name": "small_head", "description": "Small head circumference"},
                    {"name": "flat_back_head", "description": "Flattened back of head"}
                ]
            }
        }
    
    def screen_for_genetic_disorders(self, face_image: np.ndarray) -> List[GeneticScreeningResult]:
        """
        Screen face image for genetic disorders
        
        Args:
            face_image: Face image for analysis
            
        Returns:
            List[GeneticScreeningResult]: Screening results for each disorder
        """
        start_time = datetime.now()
        
        try:
            # Extract facial features
            facial_features = self._extract_facial_features(face_image)
            
            if not facial_features:
                self.logger.warning("No facial features extracted for genetic screening")
                return []
            
            results = []
            
            # Screen for each genetic disorder
            for disorder in GeneticDisorder:
                if disorder in self.models:
                    result = self._screen_for_disorder(face_image, facial_features, disorder)
                    if result:
                        results.append(result)
            
            # Sort results by risk score (highest first)
            results.sort(key=lambda x: x.risk_score, reverse=True)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update processing time for all results
            for result in results:
                result.processing_time_ms = processing_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"Genetic screening failed: {e}")
            return []
    
    def _extract_facial_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial features for genetic analysis"""
        try:
            # Detect face
            face_detection = self.face_utils.extract_largest_face(face_image)
            
            if not face_detection:
                return None
            
            # Get facial landmarks
            landmarks = face_detection.landmarks
            
            if not landmarks:
                return None
            
            # Convert landmarks to feature vector
            feature_vector = self._landmarks_to_features(landmarks)
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Failed to extract facial features: {e}")
            return None
    
    def _landmarks_to_features(self, landmarks: Dict) -> np.ndarray:
        """Convert facial landmarks to feature vector"""
        try:
            # Extract key landmark points
            features = []
            
            # Eye features
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                left_eye = landmarks['left_eye']
                right_eye = landmarks['right_eye']
                
                # Eye width and height
                left_eye_width = max([p[0] for p in left_eye]) - min([p[0] for p in left_eye])
                left_eye_height = max([p[1] for p in left_eye]) - min([p[1] for p in left_eye])
                right_eye_width = max([p[0] for p in right_eye]) - min([p[0] for p in right_eye])
                right_eye_height = max([p[1] for p in right_eye]) - min([p[1] for p in right_eye])
                
                features.extend([left_eye_width, left_eye_height, right_eye_width, right_eye_height])
                
                # Eye distance
                left_eye_center = (sum([p[0] for p in left_eye]) / len(left_eye), 
                                 sum([p[1] for p in left_eye]) / len(left_eye))
                right_eye_center = (sum([p[0] for p in right_eye]) / len(right_eye), 
                                  sum([p[1] for p in right_eye]) / len(right_eye))
                eye_distance = np.sqrt((right_eye_center[0] - left_eye_center[0])**2 + 
                                     (right_eye_center[1] - left_eye_center[1])**2)
                features.append(eye_distance)
            
            # Nose features
            if 'nose_tip' in landmarks:
                nose_tip = landmarks['nose_tip']
                nose_width = max([p[0] for p in nose_tip]) - min([p[0] for p in nose_tip])
                nose_height = max([p[1] for p in nose_tip]) - min([p[1] for p in nose_tip])
                features.extend([nose_width, nose_height])
            
            # Mouth features
            if 'top_lip' in landmarks and 'bottom_lip' in landmarks:
                top_lip = landmarks['top_lip']
                bottom_lip = landmarks['bottom_lip']
                
                mouth_width = max([p[0] for p in top_lip + bottom_lip]) - min([p[0] for p in top_lip + bottom_lip])
                mouth_height = max([p[1] for p in top_lip + bottom_lip]) - min([p[1] for p in top_lip + bottom_lip])
                features.extend([mouth_width, mouth_height])
            
            # Face shape features
            if 'chin' in landmarks:
                chin = landmarks['chin']
                chin_width = max([p[0] for p in chin]) - min([p[0] for p in chin])
                features.append(chin_width)
            
            # Pad or truncate to fixed size (68 features)
            while len(features) < 68:
                features.append(0.0)
            features = features[:68]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to convert landmarks to features: {e}")
            return np.zeros(68, dtype=np.float32)
    
    def _screen_for_disorder(self, face_image: np.ndarray, facial_features: np.ndarray, 
                           disorder: GeneticDisorder) -> Optional[GeneticScreeningResult]:
        """Screen for a specific genetic disorder"""
        try:
            if disorder not in self.models:
                return None
            
            # Get model prediction
            model = self.models[disorder]
            risk_score = model.predict(facial_features.reshape(1, -1), verbose=0)[0][0]
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_confidence(facial_features, face_image)
            
            # Extract specific facial features for this disorder
            disorder_features = self._extract_disorder_specific_features(face_image, disorder)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(disorder, risk_score, confidence)
            
            # Create result
            result = GeneticScreeningResult(
                disorder=disorder,
                risk_score=float(risk_score),
                confidence=confidence,
                facial_features=disorder_features,
                recommendation=recommendation,
                processing_time_ms=0.0,  # Will be set by caller
                disclaimer=self._get_disclaimer()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to screen for {disorder.value}: {e}")
            return None
    
    def _calculate_confidence(self, facial_features: np.ndarray, face_image: np.ndarray) -> float:
        """Calculate confidence score for the analysis"""
        try:
            # Base confidence on feature quality
            feature_quality = np.mean(np.abs(facial_features))
            
            # Adjust based on image quality
            image_quality = self.face_utils.calculate_face_quality(face_image)
            quality_score = image_quality.get("quality_score", 0.5)
            
            # Combine factors
            confidence = min(0.95, (feature_quality / 100.0 + quality_score) / 2.0)
            
            return max(0.1, confidence)  # Minimum confidence of 10%
            
        except Exception as e:
            self.logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _extract_disorder_specific_features(self, face_image: np.ndarray, 
                                          disorder: GeneticDisorder) -> List[FacialFeature]:
        """Extract disorder-specific facial features"""
        try:
            features = []
            
            if disorder.value in self.feature_definitions:
                disorder_def = self.feature_definitions[disorder.value]
                
                for feature_def in disorder_def["features"]:
                    # In a real implementation, you would extract actual measurements
                    # For now, we'll create placeholder values
                    feature = FacialFeature(
                        name=feature_def["name"],
                        value=np.random.uniform(0.0, 1.0),  # Placeholder
                        confidence=np.random.uniform(0.6, 0.9),  # Placeholder
                        description=feature_def["description"]
                    )
                    features.append(feature)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract disorder-specific features: {e}")
            return []
    
    def _generate_recommendation(self, disorder: GeneticDisorder, risk_score: float, 
                               confidence: float) -> str:
        """Generate recommendation based on screening results"""
        try:
            threshold = self.risk_thresholds.get(disorder, 0.7)
            
            if risk_score >= threshold and confidence >= 0.7:
                return f"High risk detected for {disorder.value.replace('_', ' ').title()}. " \
                       f"Recommend genetic testing and consultation with a geneticist."
            elif risk_score >= threshold * 0.7 and confidence >= 0.5:
                return f"Moderate risk detected for {disorder.value.replace('_', ' ').title()}. " \
                       f"Consider genetic testing and further evaluation."
            else:
                return f"Low risk for {disorder.value.replace('_', ' ').title()}. " \
                       f"No immediate genetic testing recommended based on facial features."
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendation: {e}")
            return "Unable to generate recommendation due to analysis error."
    
    def _get_disclaimer(self) -> str:
        """Get medical disclaimer for genetic screening"""
        return ("This is a preliminary screening tool based on facial feature analysis. "
                "It is NOT a diagnostic tool and should NOT replace professional medical evaluation. "
                "Any positive results should be confirmed through proper genetic testing and "
                "consultation with qualified healthcare professionals. This tool is for "
                "assistive screening purposes only.")
    
    def get_screening_statistics(self) -> Dict:
        """Get genetic screening statistics"""
        try:
            return {
                "supported_disorders": [disorder.value for disorder in GeneticDisorder],
                "total_models": len(self.models),
                "risk_thresholds": {disorder.value: threshold for disorder, threshold in self.risk_thresholds.items()},
                "feature_count": 68,
                "model_architecture": "Deep Neural Network with 68 facial landmark features"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get screening statistics: {e}")
            return {}
    
    def update_model(self, disorder: GeneticDisorder, new_model: tf.keras.Model) -> bool:
        """Update genetic screening model"""
        try:
            if disorder in self.models:
                self.models[disorder] = new_model
                self.logger.info(f"Updated model for {disorder.value}")
                return True
            else:
                self.logger.error(f"Unknown disorder: {disorder.value}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update model for {disorder.value}: {e}")
            return False
    
    def validate_screening_result(self, result: GeneticScreeningResult) -> bool:
        """Validate screening result"""
        try:
            # Check if risk score is in valid range
            if not (0.0 <= result.risk_score <= 1.0):
                return False
            
            # Check if confidence is in valid range
            if not (0.0 <= result.confidence <= 1.0):
                return False
            
            # Check if processing time is reasonable
            if result.processing_time_ms < 0 or result.processing_time_ms > 10000:
                return False
            
            # Check if facial features are present
            if not result.facial_features:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate screening result: {e}")
            return False
