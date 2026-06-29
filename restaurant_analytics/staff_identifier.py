import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from restaurant_analytics.interfaces import StaffIdentifier

class UniformColorIdentifier(StaffIdentifier):
    """
    Identifies staff based on the dominant uniform color of their upper body (torso).
    Uses HSV color space mask ranges.
    """
    def __init__(self, lower_hsv: Tuple[int, int, int], upper_hsv: Tuple[int, int, int], pixel_ratio_threshold: float = 0.15):
        """
        lower_hsv, upper_hsv: HSV bounds for the uniform color (e.g. red apron, green shirt, etc.)
        pixel_ratio_threshold: Minimum percentage of uniform pixels inside the torso box to classify as staff
        """
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)
        self.pixel_ratio_threshold = pixel_ratio_threshold

    def identify_staff(self, frame: np.ndarray, bbox: List[float]) -> Tuple[Optional[str], float]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = y2 - y1
        
        # Approximate torso region (middle 40% of the height, full width)
        torso_y1 = y1 + int(h * 0.25)
        torso_y2 = y1 + int(h * 0.65)
        
        # Check boundary constraints
        img_h, img_w = frame.shape[:2]
        torso_y1 = max(0, min(torso_y1, img_h - 1))
        torso_y2 = max(0, min(torso_y2, img_h - 1))
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        
        crop = frame[torso_y1:torso_y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        # Convert to HSV and apply mask
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        matching_pixels = cv2.countNonZero(mask)
        total_pixels = crop.shape[0] * crop.shape[1]
        
        ratio = matching_pixels / max(total_pixels, 1)
        
        if ratio >= self.pixel_ratio_threshold:
            # Normalize confidence score between 0.0 and 1.0
            confidence = min(1.0, ratio / (self.pixel_ratio_threshold * 2))
            return "staff_uniform", confidence
        return None, 0.0


class BadgeDetector(StaffIdentifier):
    """
    Detects a rectangular staff name badge or logo on the upper chest.
    Uses computer vision contour shape analysis (fallback) or a custom model inference if supplied.
    """
    def __init__(self, badge_model: Optional[Any] = None, confidence_threshold: float = 0.50):
        self.badge_model = badge_model
        self.confidence_threshold = confidence_threshold

    def identify_staff(self, frame: np.ndarray, bbox: List[float]) -> Tuple[Optional[str], float]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Approximate chest area (upper 20% to 45% of body box)
        chest_y1 = y1 + int((y2 - y1) * 0.20)
        chest_y2 = y1 + int((y2 - y1) * 0.45)
        
        img_h, img_w = frame.shape[:2]
        chest_y1 = max(0, min(chest_y1, img_h - 1))
        chest_y2 = max(0, min(chest_y2, img_h - 1))
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        
        crop = frame[chest_y1:chest_y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        if self.badge_model is not None:
            # Placeholder for running a custom YOLO/Detectron model specifically trained on badges
            results = self.badge_model(crop, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                conf = float(results[0].boxes[0].conf[0])
                if conf >= self.confidence_threshold:
                    return "staff_badge", conf
        else:
            # CV Heuristic: Search for high-contrast badge-like shapes (white/yellow rectangular tags)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h) if h > 0 else 0
                # A typical name badge is rectangular (e.g. aspect ratio 1.5 to 3.0) and small
                if 1.2 <= aspect_ratio <= 3.5 and 15 <= w <= 80 and 8 <= h <= 45:
                    return "staff_badge_cv", 0.70  # CV heuristic confidence constant
                    
        return None, 0.0


class EnrolledEmbeddingMatcher(StaffIdentifier):
    """
    Compares person Re-ID appearance embeddings against pre-enrolled employee embeddings.
    """
    def __init__(self, enrolled_embeddings: Dict[str, np.ndarray], similarity_threshold: float = 0.75):
        """
        enrolled_embeddings: dict mapping staff_id -> 1D numpy array Re-ID embedding vector
        similarity_threshold: Cosine similarity threshold to match staff
        """
        self.enrolled = enrolled_embeddings
        self.similarity_threshold = similarity_threshold

    def identify_staff(self, frame: np.ndarray, bbox: List[float]) -> Tuple[Optional[str], float]:
        # Note: BBox is passed, but this class expects an embedding vector.
        # In a multimodal system, the edge agent generates the Re-ID embedding and passes it.
        # We can accept an embedding via metadata or calculate it if the feature extractor is present.
        return None, 0.0

    def match_embedding(self, query_emb: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Compares query embedding with enrolled staff embeddings.
        """
        best_id = None
        best_score = 0.0
        
        # Ensure query is unit normalized
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        for staff_id, stored_emb in self.enrolled.items():
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
            score = float(np.dot(query_norm, stored_norm))
            if score > best_score:
                best_score = score
                best_id = staff_id
                
        if best_score >= self.similarity_threshold:
            return best_id, best_score
        return None, 0.0


class MultiModalStaffIdentifier(StaffIdentifier):
    """
    Combines Uniform, Badge, and Re-ID Embedding matches using an ensemble heuristic.
    """
    def __init__(
        self,
        color_identifier: Optional[UniformColorIdentifier] = None,
        badge_detector: Optional[BadgeDetector] = None,
        embedding_matcher: Optional[EnrolledEmbeddingMatcher] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.color_id = color_identifier
        self.badge_id = badge_detector
        self.emb_id = embedding_matcher
        self.weights = weights or {"color": 0.4, "badge": 0.5, "embedding": 0.6}

    def identify_staff(self, frame: np.ndarray, bbox: List[float]) -> Tuple[Optional[str], float]:
        """
        Checks all active identifier modules and scores the result.
        """
        scores: List[Tuple[str, float]] = []
        
        # 1. Color Classifier
        if self.color_id:
            label, score = self.color_id.identify_staff(frame, bbox)
            if label:
                scores.append((label, score * self.weights["color"]))
                
        # 2. Badge Detector
        if self.badge_id:
            label, score = self.badge_id.identify_staff(frame, bbox)
            if label:
                scores.append((label, score * self.weights["badge"]))

        # If we have positive hits from either physical visual cues (apron/uniform OR badge)
        if scores:
            best_label, weighted_score = max(scores, key=lambda x: x[1])
            # Return label with aggregated score
            return best_label, min(1.0, weighted_score)
            
        return None, 0.0

    def identify_with_embedding(
        self,
        frame: np.ndarray,
        bbox: List[float],
        embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Combines standard bbox visuals with Re-ID embedding identity matching.
        """
        # First check Re-ID database
        if self.emb_id:
            staff_id, sim = self.emb_id.match_embedding(embedding)
            if staff_id:
                # If they match a registered face/appearance database, trust it with high weight
                return staff_id, min(1.0, sim * self.weights["embedding"])

        # Fallback to physical cues
        return self.identify_staff(frame, bbox)
