"""
Face Utilities for Healthcare Face Recognition System
Common face processing utilities and helper functions
"""
import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass


@dataclass
class FaceDetection:
    """Face detection result"""
    face_location: Tuple[int, int, int, int]  # top, right, bottom, left
    face_encoding: np.ndarray
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None


class FaceUtils:
    """
    Utility class for face processing operations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image
        
        Args:
            image: Input image
            
        Returns:
            List[FaceDetection]: List of detected faces
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            # Get face landmarks
            face_landmarks = face_recognition.face_landmarks(rgb_image, face_locations)
            
            detections = []
            for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
                landmarks = face_landmarks[i] if i < len(face_landmarks) else None
                
                detection = FaceDetection(
                    face_location=location,
                    face_encoding=encoding,
                    confidence=1.0,  # face_recognition doesn't provide confidence
                    landmarks=landmarks
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_largest_face(self, image: np.ndarray) -> Optional[FaceDetection]:
        """
        Extract the largest face from an image
        
        Args:
            image: Input image
            
        Returns:
            Optional[FaceDetection]: Largest face detection or None
        """
        try:
            detections = self.detect_faces(image)
            
            if not detections:
                return None
            
            # Find largest face by area
            largest_detection = max(detections, key=lambda d: self._calculate_face_area(d.face_location))
            
            return largest_detection
            
        except Exception as e:
            self.logger.error(f"Failed to extract largest face: {e}")
            return None
    
    def _calculate_face_area(self, face_location: Tuple[int, int, int, int]) -> int:
        """Calculate face area from location"""
        top, right, bottom, left = face_location
        return (bottom - top) * (right - left)
    
    def crop_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int], 
                  padding: float = 0.2) -> np.ndarray:
        """
        Crop face from image with padding
        
        Args:
            image: Input image
            face_location: Face location (top, right, bottom, left)
            padding: Padding factor (0.2 = 20% padding)
            
        Returns:
            np.ndarray: Cropped face image
        """
        try:
            top, right, bottom, left = face_location
            
            # Calculate padding
            height = bottom - top
            width = right - left
            
            pad_top = int(height * padding)
            pad_bottom = int(height * padding)
            pad_left = int(width * padding)
            pad_right = int(width * padding)
            
            # Apply padding
            top = max(0, top - pad_top)
            bottom = min(image.shape[0], bottom + pad_bottom)
            left = max(0, left - pad_left)
            right = min(image.shape[1], right + pad_right)
            
            # Crop face
            cropped_face = image[top:bottom, left:right]
            
            return cropped_face
            
        except Exception as e:
            self.logger.error(f"Failed to crop face: {e}")
            return image
    
    def normalize_face(self, face_image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Normalize face image for processing
        
        Args:
            face_image: Face image
            target_size: Target size (width, height)
            
        Returns:
            np.ndarray: Normalized face image
        """
        try:
            # Resize to target size
            resized = cv2.resize(face_image, target_size)
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Failed to normalize face: {e}")
            return face_image
    
    def calculate_face_quality(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Calculate face image quality metrics
        
        Args:
            face_image: Face image
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            
            # Calculate various quality metrics
            metrics = {}
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics["sharpness"] = laplacian_var
            
            # 2. Brightness
            brightness = np.mean(gray)
            metrics["brightness"] = brightness
            
            # 3. Contrast
            contrast = np.std(gray)
            metrics["contrast"] = contrast
            
            # 4. Blur detection
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics["blur_score"] = blur_score
            
            # 5. Overall quality score (0-1)
            quality_score = min(1.0, (laplacian_var / 1000.0 + contrast / 100.0) / 2.0)
            metrics["quality_score"] = quality_score
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate face quality: {e}")
            return {"quality_score": 0.0}
    
    def detect_face_orientation(self, face_image: np.ndarray) -> str:
        """
        Detect face orientation
        
        Args:
            face_image: Face image
            
        Returns:
            str: Orientation ('front', 'left', 'right', 'up', 'down')
        """
        try:
            # This is a simplified implementation
            # In practice, you would use a more sophisticated method
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            
            # Detect eyes
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(eyes) >= 2:
                # Calculate eye positions
                eye_centers = []
                for (ex, ey, ew, eh) in eyes:
                    center_x = ex + ew // 2
                    center_y = ey + eh // 2
                    eye_centers.append((center_x, center_y))
                
                if len(eye_centers) >= 2:
                    # Calculate eye line angle
                    eye1, eye2 = eye_centers[0], eye_centers[1]
                    angle = np.arctan2(eye2[1] - eye1[1], eye2[0] - eye1[0]) * 180 / np.pi
                    
                    if abs(angle) < 10:
                        return "front"
                    elif angle > 10:
                        return "left"
                    else:
                        return "right"
            
            return "unknown"
            
        except Exception as e:
            self.logger.error(f"Failed to detect face orientation: {e}")
            return "unknown"
    
    def enhance_face_image(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance face image quality
        
        Args:
            face_image: Face image
            
        Returns:
            np.ndarray: Enhanced face image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply bilateral filter for noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Failed to enhance face image: {e}")
            return face_image
    
    def validate_face_image(self, face_image: np.ndarray) -> Dict[str, bool]:
        """
        Validate face image for processing
        
        Args:
            face_image: Face image
            
        Returns:
            Dict[str, bool]: Validation results
        """
        try:
            validation = {
                "has_face": False,
                "good_quality": False,
                "proper_size": False,
                "good_lighting": False,
                "front_facing": False
            }
            
            # Check if image has a face
            detections = self.detect_faces(face_image)
            validation["has_face"] = len(detections) > 0
            
            if not validation["has_face"]:
                return validation
            
            # Get quality metrics
            quality_metrics = self.calculate_face_quality(face_image)
            
            # Check quality
            validation["good_quality"] = quality_metrics.get("quality_score", 0) > 0.5
            
            # Check size
            height, width = face_image.shape[:2]
            validation["proper_size"] = height >= 100 and width >= 100
            
            # Check lighting
            brightness = quality_metrics.get("brightness", 0)
            validation["good_lighting"] = 50 < brightness < 200
            
            # Check orientation
            orientation = self.detect_face_orientation(face_image)
            validation["front_facing"] = orientation == "front"
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Failed to validate face image: {e}")
            return {"error": True}
    
    def create_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Create face embedding from image
        
        Args:
            face_image: Face image
            
        Returns:
            Optional[np.ndarray]: Face embedding or None
        """
        try:
            # Extract largest face
            detection = self.extract_largest_face(face_image)
            
            if detection is None:
                return None
            
            return detection.face_encoding
            
        except Exception as e:
            self.logger.error(f"Failed to create face embedding: {e}")
            return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                     threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Compare two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Comparison threshold
            
        Returns:
            Tuple[bool, float]: (match, distance)
        """
        try:
            # Calculate face distance
            distance = face_recognition.face_distance([embedding1], embedding2)[0]
            
            # Convert distance to similarity score
            similarity = 1 - distance
            
            # Check if faces match
            match = similarity > (1 - threshold)
            
            return match, similarity
            
        except Exception as e:
            self.logger.error(f"Failed to compare faces: {e}")
            return False, 0.0
