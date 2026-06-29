import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Generator, Tuple, Optional
import cv2
import numpy as np

from restaurant_analytics.schema import RestaurantEvent
from restaurant_analytics.interfaces import (
    PersonDetector, MultiObjectTracker, ActivityClassifier, StaffIdentifier
)
from restaurant_analytics.zone_mapper import ZoneMapper

class RestaurantEdgeAgent:
    """
    Main orchestrator running at the edge. Captures video frames,
    runs detectors/trackers, classifies activities, and emits schema-compliant JSON events.
    """
    def __init__(
        self,
        camera_id: str,
        detector: PersonDetector,
        tracker: MultiObjectTracker,
        zone_mapper: ZoneMapper,
        activity_classifier: Optional[ActivityClassifier] = None,
        staff_identifier: Optional[StaffIdentifier] = None,
        enable_anon: bool = True
    ):
        self.camera_id = camera_id
        self.detector = detector
        self.tracker = tracker
        self.zone_mapper = zone_mapper
        self.activity_classifier = activity_classifier
        self.staff_identifier = staff_identifier
        self.enable_anon = enable_anon
        
        # State tracking: track_id -> current state
        self.active_tracks: Dict[str, Dict[str, Any]] = {}

    def anonymize_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Applies face blurring or anonymization to detections.
        For skeleton purposes, blurs the upper portion of person bounding boxes (head/face area).
        """
        if not self.enable_anon:
            return frame
            
        anonymized = frame.copy()
        for det in detections:
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # Face area approximation: upper 25% of the person's bounding box
                face_height = int((y2 - y1) * 0.25)
                if face_height > 0:
                    face_roi = anonymized[y1:y1+face_height, x1:x2]
                    if face_roi.size > 0:
                        # Apply Gaussian blur to the face region
                        blurred = cv2.GaussianBlur(face_roi, (15, 15), 0)
                        anonymized[y1:y1+face_height, x1:x2] = blurred
        return anonymized

    def _generate_event(
        self,
        track_id: str,
        event_type: str,
        zone_id: Optional[str],
        confidence: float,
        meta: Dict[str, Any]
    ) -> RestaurantEvent:
        """
        Helper to construct a standard RestaurantEvent.
        """
        return RestaurantEvent(
            event_id=str(uuid.uuid4()),
            camera_id=self.camera_id,
            track_id=track_id,
            zone_id=zone_id,
            event_type=event_type,
            start_ts=datetime.now(timezone.utc).isoformat(),
            end_ts=None,
            confidence=confidence,
            meta=meta
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        ts: float
    ) -> Generator[RestaurantEvent, None, None]:
        """
        Processes a single frame, updates tracking and zones, and yields new events.
        """
        # 1. Run detection
        detections = self.detector.detect(frame)

        # 2. Anonymize if enabled
        if self.enable_anon:
            frame = self.anonymize_frame(frame, detections)

        # 3. Format detections for tracking: [x1, y1, x2, y2, confidence]
        tracker_inputs = []
        for d in detections:
            bbox = d["bbox"]
            conf = d.get("confidence", 1.0)
            tracker_inputs.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])

        # 4. Update tracker
        tracks = self.tracker.update(tracker_inputs, frame_id, ts)
        
        current_active_track_ids = set()

        for track_id, bbox, is_new in tracks:
            current_active_track_ids.add(track_id)
            
            # Map tracking box to floor plan zone
            zone_id = self.zone_mapper.get_zone_for_bbox(bbox)
            
            # Identify if staff
            staff_id = None
            staff_conf = 0.0
            if self.staff_identifier:
                staff_id, staff_conf = self.staff_identifier.identify_staff(frame, bbox)

            # Classify activity
            activity_label = "unknown"
            activity_conf = 0.0
            if self.activity_classifier:
                # Stub out simple history parameter
                activity_label, activity_conf = self.activity_classifier.classify_activity(frame, bbox, [])

            # Get track state
            if is_new or (track_id not in self.active_tracks):
                self.active_tracks[track_id] = {
                    "last_zone": None,
                    "last_activity": None,
                    "is_staff": staff_id is not None,
                    "staff_id": staff_id,
                    "first_seen": ts
                }
                
                # Emit registration event
                yield self._generate_event(
                    track_id=track_id,
                    event_type="register_person",
                    zone_id=zone_id,
                    confidence=1.0,
                    meta={"bbox": bbox, "is_staff": staff_id is not None, "staff_id": staff_id}
                )

            track_state = self.active_tracks[track_id]

            # Detect zone transitions
            if zone_id != track_state["last_zone"]:
                if track_state["last_zone"] is not None:
                    # Emit exit event for old zone
                    yield self._generate_event(
                        track_id=track_id,
                        event_type="exit_zone",
                        zone_id=track_state["last_zone"],
                        confidence=0.95,
                        meta={"bbox": bbox}
                    )
                if zone_id is not None:
                    # Emit enter event for new zone
                    yield self._generate_event(
                        track_id=track_id,
                        event_type="enter_zone",
                        zone_id=zone_id,
                        confidence=0.95,
                        meta={"bbox": bbox}
                    )
                track_state["last_zone"] = zone_id

            # Detect activity changes
            if activity_label != track_state["last_activity"] and activity_label != "unknown":
                if track_state["last_activity"] is not None:
                    yield self._generate_event(
                        track_id=track_id,
                        event_type="activity_end",
                        zone_id=zone_id,
                        confidence=activity_conf,
                        meta={"activity": track_state["last_activity"]}
                    )
                yield self._generate_event(
                    track_id=track_id,
                    event_type="activity_start",
                    zone_id=zone_id,
                    confidence=activity_conf,
                    meta={"activity": activity_label}
                )
                track_state["last_activity"] = activity_label

        # Handle exit detection for track IDs no longer active
        # (In production, the tracker handles pruning; we align state mappings here)
        for old_track_id in list(self.active_tracks.keys()):
            if old_track_id not in current_active_track_ids:
                # If they were in a zone, emit final zone exit
                last_zone = self.active_tracks[old_track_id]["last_zone"]
                if last_zone is not None:
                    yield self._generate_event(
                        track_id=old_track_id,
                        event_type="exit_zone",
                        zone_id=last_zone,
                        confidence=1.0,
                        meta={}
                    )
                # Emit deregistration
                yield self._generate_event(
                    track_id=old_track_id,
                    event_type="deregister_person",
                    zone_id=None,
                    confidence=1.0,
                    meta={}
                )
                del self.active_tracks[old_track_id]
