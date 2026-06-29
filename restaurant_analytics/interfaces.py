from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class PersonDetector(ABC):
    """
    Interface for person and object detection models.
    """
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on the input frame.
        Returns:
            List of dicts: {"bbox": [x1, y1, x2, y2], "class_id": int, "confidence": float}
        """
        pass

class MultiObjectTracker(ABC):
    """
    Interface for tracking entities across video frames.
    """
    @abstractmethod
    def update(self, detections: List[List[float]], frame_id: int, ts: float) -> List[Tuple[str, List[float], bool]]:
        """
        Updates tracker state with new bounding boxes.
        Returns:
            List of tuples: (track_id, bbox, is_new)
        """
        pass

class ActivityClassifier(ABC):
    """
    Interface for classifying actions/poses (e.g. eating, sitting, waiting).
    """
    @abstractmethod
    def classify_activity(self, frame: np.ndarray, bbox: List[float], history: List[np.ndarray]) -> Tuple[str, float]:
        """
        Classifies activity of a person inside the bounding box based on current frame and past context.
        Returns:
            Tuple of (activity_label, confidence)
        """
        pass

class StaffIdentifier(ABC):
    """
    Interface for distinguishing staff/employees from customers.
    """
    @abstractmethod
    def identify_staff(self, frame: np.ndarray, bbox: List[float]) -> Tuple[Optional[str], float]:
        """
        Examines person crop for staff badge, uniform, or face embedding.
        Returns:
            Tuple of (staff_id or None if customer, confidence)
        """
        pass
