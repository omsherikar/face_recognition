"""
Anti-Spoofing Security Module for Healthcare Face Recognition System
Implements liveness detection and spoofing prevention
"""
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque

from config.settings import settings


class LivenessMethod(Enum):
    """Liveness detection methods"""
    BLINK_DETECTION = "blink_detection"
    HEAD_MOVEMENT = "head_movement"
    DEPTH_ANALYSIS = "depth_analysis"
    TEXTURE_ANALYSIS = "texture_analysis"
    MULTI_FRAME = "multi_frame"
    INFRARED = "infrared"


@dataclass
class LivenessResult:
    """Result of liveness detection"""
    is_live: bool
    confidence: float
    method_used: LivenessMethod
    processing_time_ms: float
    details: Dict


class AntiSpoofingDetector:
    """
    Anti-spoofing detector using multiple liveness detection methods
    Implements state-of-the-art spoofing detection techniques
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Frame history for temporal analysis
        self.frame_history: Dict[str, deque] = {}
        self.blink_history: Dict[str, deque] = {}
        
        # Load liveness detection model (placeholder)
        self.liveness_model = self._load_liveness_model()
        
        # Configuration
        self.min_frames_for_analysis = 5
        self.blink_threshold = 0.3
        self.movement_threshold = 10.0
        
    def _load_liveness_model(self):
        """Load pre-trained liveness detection model"""
        try:
            # In a real implementation, you would load a trained model
            # For now, we'll use a placeholder
            self.logger.info("Liveness detection model loaded (placeholder)")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load liveness model: {e}")
            return None
    
    def is_live_face(self, image: np.ndarray, session_id: str = "default") -> LivenessResult:
        """
        Comprehensive liveness detection using multiple methods
        
        Args:
            image: Input image
            session_id: Session identifier for tracking
            
        Returns:
            LivenessResult: Liveness detection result
        """
        start_time = time.time()
        
        try:
            # Initialize session tracking if needed
            if session_id not in self.frame_history:
                self.frame_history[session_id] = deque(maxlen=30)
                self.blink_history[session_id] = deque(maxlen=10)
            
            # Store current frame
            self.frame_history[session_id].append(image.copy())
            
            # Run multiple liveness detection methods
            results = []
            
            # 1. Blink detection
            blink_result = self._detect_blinks(image, session_id)
            if blink_result:
                results.append(blink_result)
            
            # 2. Head movement detection
            movement_result = self._detect_head_movement(session_id)
            if movement_result:
                results.append(movement_result)
            
            # 3. Texture analysis
            texture_result = self._analyze_texture(image)
            if texture_result:
                results.append(texture_result)
            
            # 4. Multi-frame analysis
            if len(self.frame_history[session_id]) >= self.min_frames_for_analysis:
                multi_frame_result = self._multi_frame_analysis(session_id)
                if multi_frame_result:
                    results.append(multi_frame_result)
            
            # Combine results
            final_result = self._combine_liveness_results(results)
            
            processing_time = (time.time() - start_time) * 1000
            
            return LivenessResult(
                is_live=final_result["is_live"],
                confidence=final_result["confidence"],
                method_used=LivenessMethod.MULTI_FRAME,
                processing_time_ms=processing_time,
                details=final_result["details"]
            )
            
        except Exception as e:
            self.logger.error(f"Liveness detection failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                method_used=LivenessMethod.MULTI_FRAME,
                processing_time_ms=processing_time,
                details={"error": str(e)}
            )
    
    def _detect_blinks(self, image: np.ndarray, session_id: str) -> Optional[Dict]:
        """Detect eye blinks for liveness"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(eyes) >= 2:
                # Calculate eye aspect ratio
                eye_ratios = []
                for (ex, ey, ew, eh) in eyes:
                    eye_region = gray[ey:ey+eh, ex:ex+ew]
                    eye_ratio = self._calculate_eye_aspect_ratio(eye_region)
                    eye_ratios.append(eye_ratio)
                
                avg_eye_ratio = np.mean(eye_ratios)
                self.blink_history[session_id].append(avg_eye_ratio)
                
                # Detect blink pattern
                if len(self.blink_history[session_id]) >= 3:
                    blink_detected = self._analyze_blink_pattern(session_id)
                    
                    return {
                        "method": "blink_detection",
                        "is_live": blink_detected,
                        "confidence": 0.8 if blink_detected else 0.2,
                        "eye_ratio": avg_eye_ratio,
                        "blink_count": len([r for r in self.blink_history[session_id] if r < self.blink_threshold])
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Blink detection failed: {e}")
            return None
    
    def _calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """Calculate eye aspect ratio"""
        try:
            # Find contours in eye region
            contours, _ = cv2.findContours(eye_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate aspect ratio
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h
                
                return aspect_ratio
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Eye aspect ratio calculation failed: {e}")
            return 1.0
    
    def _analyze_blink_pattern(self, session_id: str) -> bool:
        """Analyze blink pattern for liveness"""
        try:
            ratios = list(self.blink_history[session_id])
            
            # Look for blink pattern (eye closure and opening)
            blinks = 0
            for i in range(1, len(ratios) - 1):
                if ratios[i] < self.blink_threshold and ratios[i-1] >= self.blink_threshold and ratios[i+1] >= self.blink_threshold:
                    blinks += 1
            
            # Consider live if at least one blink detected
            return blinks > 0
            
        except Exception as e:
            self.logger.error(f"Blink pattern analysis failed: {e}")
            return False
    
    def _detect_head_movement(self, session_id: str) -> Optional[Dict]:
        """Detect head movement for liveness"""
        try:
            if len(self.frame_history[session_id]) < 2:
                return None
            
            # Get face positions from recent frames
            face_positions = []
            for frame in list(self.frame_history[session_id])[-5:]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Get center of largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    center_x = largest_face[0] + largest_face[2] // 2
                    center_y = largest_face[1] + largest_face[3] // 2
                    face_positions.append((center_x, center_y))
            
            if len(face_positions) >= 2:
                # Calculate movement
                movements = []
                for i in range(1, len(face_positions)):
                    dx = face_positions[i][0] - face_positions[i-1][0]
                    dy = face_positions[i][1] - face_positions[i-1][1]
                    movement = np.sqrt(dx*dx + dy*dy)
                    movements.append(movement)
                
                avg_movement = np.mean(movements)
                max_movement = np.max(movements)
                
                # Consider live if there's significant movement
                is_live = avg_movement > self.movement_threshold or max_movement > self.movement_threshold * 2
                
                return {
                    "method": "head_movement",
                    "is_live": is_live,
                    "confidence": min(0.9, avg_movement / 50.0),
                    "avg_movement": avg_movement,
                    "max_movement": max_movement
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Head movement detection failed: {e}")
            return None
    
    def _analyze_texture(self, image: np.ndarray) -> Optional[Dict]:
        """Analyze texture for spoofing detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_region = gray[y:y+h, x:x+w]
                
                # Calculate texture features
                # 1. Local Binary Pattern variance
                lbp_variance = self._calculate_lbp_variance(face_region)
                
                # 2. Gradient magnitude
                gradient_magnitude = self._calculate_gradient_magnitude(face_region)
                
                # 3. Laplacian variance
                laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
                
                # Combine features for liveness score
                texture_score = (lbp_variance + gradient_magnitude + laplacian_var) / 3.0
                
                # Threshold for liveness (higher texture usually means real face)
                is_live = texture_score > 100.0
                confidence = min(0.9, texture_score / 200.0)
                
                return {
                    "method": "texture_analysis",
                    "is_live": is_live,
                    "confidence": confidence,
                    "texture_score": texture_score,
                    "lbp_variance": lbp_variance,
                    "gradient_magnitude": gradient_magnitude,
                    "laplacian_variance": laplacian_var
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Texture analysis failed: {e}")
            return None
    
    def _calculate_lbp_variance(self, image: np.ndarray) -> float:
        """Calculate Local Binary Pattern variance"""
        try:
            # Simplified LBP calculation
            rows, cols = image.shape
            lbp_image = np.zeros_like(image)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = image[i, j]
                    binary_string = ""
                    
                    # 8-neighborhood
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += "1" if neighbor >= center else "0"
                    
                    lbp_image[i, j] = int(binary_string, 2)
            
            return lbp_image.var()
            
        except Exception as e:
            self.logger.error(f"LBP variance calculation failed: {e}")
            return 0.0
    
    def _calculate_gradient_magnitude(self, image: np.ndarray) -> float:
        """Calculate gradient magnitude"""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return magnitude.mean()
            
        except Exception as e:
            self.logger.error(f"Gradient magnitude calculation failed: {e}")
            return 0.0
    
    def _multi_frame_analysis(self, session_id: str) -> Optional[Dict]:
        """Multi-frame temporal analysis"""
        try:
            frames = list(self.frame_history[session_id])
            
            if len(frames) < self.min_frames_for_analysis:
                return None
            
            # Calculate frame differences
            frame_diffs = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                diff_mean = np.mean(diff)
                frame_diffs.append(diff_mean)
            
            # Analyze temporal consistency
            avg_diff = np.mean(frame_diffs)
            diff_variance = np.var(frame_diffs)
            
            # Real faces should have consistent but not too high frame differences
            is_live = 5.0 < avg_diff < 50.0 and diff_variance < 100.0
            
            confidence = min(0.9, avg_diff / 30.0)
            
            return {
                "method": "multi_frame",
                "is_live": is_live,
                "confidence": confidence,
                "avg_frame_diff": avg_diff,
                "diff_variance": diff_variance,
                "frame_count": len(frames)
            }
            
        except Exception as e:
            self.logger.error(f"Multi-frame analysis failed: {e}")
            return None
    
    def _combine_liveness_results(self, results: List[Dict]) -> Dict:
        """Combine multiple liveness detection results"""
        try:
            if not results:
                return {
                    "is_live": False,
                    "confidence": 0.0,
                    "details": {"error": "No liveness detection results"}
                }
            
            # Weighted combination of results
            total_confidence = 0.0
            total_weight = 0.0
            live_votes = 0
            
            method_weights = {
                "blink_detection": 0.4,
                "head_movement": 0.3,
                "texture_analysis": 0.2,
                "multi_frame": 0.1
            }
            
            details = {}
            
            for result in results:
                method = result["method"]
                weight = method_weights.get(method, 0.1)
                
                total_confidence += result["confidence"] * weight
                total_weight += weight
                
                if result["is_live"]:
                    live_votes += 1
                
                details[method] = result
            
            # Final decision
            final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
            is_live = live_votes > len(results) / 2 and final_confidence > 0.5
            
            return {
                "is_live": is_live,
                "confidence": final_confidence,
                "details": details,
                "live_votes": live_votes,
                "total_methods": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to combine liveness results: {e}")
            return {
                "is_live": False,
                "confidence": 0.0,
                "details": {"error": str(e)}
            }
    
    def detect_spoofing_attempts(self, session_id: str) -> Dict:
        """Detect potential spoofing attempts"""
        try:
            attempts = {
                "static_image": False,
                "video_replay": False,
                "mask_detection": False,
                "screen_detection": False,
                "total_attempts": 0
            }
            
            if session_id not in self.frame_history:
                return attempts
            
            frames = list(self.frame_history[session_id])
            
            if len(frames) < 3:
                return attempts
            
            # Check for static image (no movement)
            frame_diffs = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                diff_mean = np.mean(diff)
                frame_diffs.append(diff_mean)
            
            if np.mean(frame_diffs) < 2.0:
                attempts["static_image"] = True
                attempts["total_attempts"] += 1
            
            # Check for video replay (periodic patterns)
            if len(frame_diffs) > 10:
                # Look for periodic patterns in frame differences
                fft = np.fft.fft(frame_diffs)
                frequencies = np.fft.fftfreq(len(frame_diffs))
                
                # Check for strong periodic components
                power_spectrum = np.abs(fft)**2
                if np.max(power_spectrum[1:]) > np.mean(power_spectrum) * 3:
                    attempts["video_replay"] = True
                    attempts["total_attempts"] += 1
            
            return attempts
            
        except Exception as e:
            self.logger.error(f"Failed to detect spoofing attempts: {e}")
            return {"error": str(e)}
    
    def clear_session_data(self, session_id: str):
        """Clear session data for privacy"""
        try:
            if session_id in self.frame_history:
                del self.frame_history[session_id]
            if session_id in self.blink_history:
                del self.blink_history[session_id]
            
            self.logger.info(f"Cleared session data for {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to clear session data for {session_id}: {e}")
    
    def get_security_metrics(self) -> Dict:
        """Get security metrics"""
        try:
            return {
                "active_sessions": len(self.frame_history),
                "total_frames_processed": sum(len(frames) for frames in self.frame_history.values()),
                "liveness_methods": [method.value for method in LivenessMethod],
                "detection_thresholds": {
                    "blink_threshold": self.blink_threshold,
                    "movement_threshold": self.movement_threshold,
                    "min_frames_for_analysis": self.min_frames_for_analysis
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get security metrics: {e}")
            return {}
