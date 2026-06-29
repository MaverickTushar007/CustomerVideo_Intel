import unittest
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from restaurant_analytics.schema import RestaurantEvent
from restaurant_analytics.interfaces import (
    PersonDetector, MultiObjectTracker, ActivityClassifier, StaffIdentifier
)
from restaurant_analytics.zone_mapper import ZoneMapper
from restaurant_analytics.edge_agent import RestaurantEdgeAgent
from restaurant_analytics.staff_identifier import (
    UniformColorIdentifier, BadgeDetector, EnrolledEmbeddingMatcher, MultiModalStaffIdentifier
)

# ----------------- Stub Implementations for Testing -----------------

class DummyDetector(PersonDetector):
    def __init__(self, detections: List[Dict[str, Any]]):
        self.detections = detections

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return self.detections

class DummyTracker(MultiObjectTracker):
    def __init__(self, tracks: List[Tuple[str, List[float], bool]]):
        self.tracks = tracks

    def update(self, detections: List[List[float]], frame_id: int, ts: float) -> List[Tuple[str, List[float], bool]]:
        return self.tracks

class DummyActivityClassifier(ActivityClassifier):
    def __init__(self, activity: str, confidence: float):
        self.activity = activity
        self.confidence = confidence

    def classify_activity(self, frame: np.ndarray, bbox: List[float], history: List[np.ndarray]) -> Tuple[str, float]:
        return self.activity, self.confidence

class DummyStaffIdentifier(StaffIdentifier):
    def __init__(self, staff_id: Optional[str], confidence: float):
        self.staff_id = staff_id
        self.confidence = confidence

    def identify_staff(self, frame: np.ndarray, bbox: List[float]) -> Tuple[Optional[str], float]:
        return self.staff_id, self.confidence

# ----------------- Unit Tests -----------------

class TestRestaurantAnalytics(unittest.TestCase):

    def test_restaurant_event_schema(self):
        """
        Verifies that the RestaurantEvent validation schema behaves correctly.
        """
        event = RestaurantEvent(
            event_id="test-uuid-1234",
            camera_id="cam_01",
            track_id="trk_999",
            zone_id="table_3",
            event_type="enter_zone",
            start_ts="2026-06-30T00:00:00Z",
            confidence=0.98,
            meta={"lighting": "normal"}
        )
        
        self.assertEqual(event.event_id, "test-uuid-1234")
        self.assertEqual(event.camera_id, "cam_01")
        self.assertEqual(event.zone_id, "table_3")
        self.assertEqual(event.meta["lighting"], "normal")

    def test_zone_mapper_containment(self):
        """
        Verifies that ZoneMapper detects containment inside polygon zones.
        """
        # Simple unit square table zone
        zones = {
            "table_1": [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        }
        mapper = ZoneMapper(zones=zones)
        
        # Bottom-center of bbox inside unit square
        bbox_inside = [1.0, 1.0, 5.0, 5.0]  # bottom center = (3.0, 5.0)
        self.assertEqual(mapper.get_zone_for_bbox(bbox_inside), "table_1")
        
        # Bottom-center of bbox outside
        bbox_outside = [12.0, 1.0, 15.0, 5.0]  # bottom center = (13.5, 5.0)
        self.assertIsNone(mapper.get_zone_for_bbox(bbox_outside))

    def test_edge_agent_pipeline(self):
        """
        Tests the main Edge Agent processing loop and validates emitted event flow.
        """
        # Set up mocks
        detector = DummyDetector([{"bbox": [2.0, 2.0, 8.0, 8.0], "confidence": 0.95}])
        tracker = DummyTracker([("trk_1", [2.0, 2.0, 8.0, 8.0], True)])
        
        zones = {
            "dining_area": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
        }
        zone_mapper = ZoneMapper(zones=zones)
        
        activity_classifier = DummyActivityClassifier("eating", 0.88)
        staff_identifier = DummyStaffIdentifier(None, 1.0) # customer

        agent = RestaurantEdgeAgent(
            camera_id="cam_test_01",
            detector=detector,
            tracker=tracker,
            zone_mapper=zone_mapper,
            activity_classifier=activity_classifier,
            staff_identifier=staff_identifier,
            enable_anon=False
        )
        
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Run one frame
        events = list(agent.process_frame(dummy_frame, frame_id=1, ts=0.125))
        
        # Expect:
        # 1. register_person event
        # 2. enter_zone event (dining_area)
        # 3. activity_start event (eating)
        self.assertEqual(len(events), 3)
        
        event_types = [e.event_type for e in events]
        self.assertIn("register_person", event_types)
        self.assertIn("enter_zone", event_types)
        self.assertIn("activity_start", event_types)
        
        # Validate structure compatibility
        for e in events:
            self.assertIsInstance(e, RestaurantEvent)
            self.assertEqual(e.camera_id, "cam_test_01")

    def test_staff_identifiers(self):
        """
        Validates the 3 staff classification methods (Uniform, Badge, and Re-ID embedding) and fusion.
        """
        bbox = [0.0, 0.0, 50.0, 100.0]
        
        # 1. Test UniformColorIdentifier
        # Create a mock frame where the torso is solid red (Hue: ~0, Saturation: 255, Value: 255)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        # BGR red is (0, 0, 255)
        frame[25:65, 0:50] = (0, 0, 255)
        
        # Red HSV bounds in OpenCV are around Hue 0-10 or 170-180.
        # Let's specify bounds that capture our BGR red (HSV = [0, 255, 255])
        lower_red = (0, 100, 100)
        upper_red = (10, 255, 255)
        color_id = UniformColorIdentifier(lower_hsv=lower_red, upper_hsv=upper_red, pixel_ratio_threshold=0.15)
        
        label, conf = color_id.identify_staff(frame, bbox)
        self.assertEqual(label, "staff_uniform")
        self.assertGreater(conf, 0.5)

        # 2. Test EnrolledEmbeddingMatcher
        staff_db = {
            "emp_jack": np.array([0.5, 0.5, 0.5, 0.5]),
            "emp_mary": np.array([1.0, 0.0, 0.0, 0.0])
        }
        emb_matcher = EnrolledEmbeddingMatcher(enrolled_embeddings=staff_db, similarity_threshold=0.85)
        
        # Perfect match queryMary
        query_mary = np.array([1.0, 0.0, 0.0, 0.0])
        staff_id, score = emb_matcher.match_embedding(query_mary)
        self.assertEqual(staff_id, "emp_mary")
        self.assertAlmostEqual(score, 1.0)
        
        # Non-matching query
        query_none = np.array([0.0, 1.0, 0.0, 0.0])
        staff_id, score = emb_matcher.match_embedding(query_none)
        self.assertIsNone(staff_id)
        self.assertEqual(score, 0.0)

        # 3. Test BadgeDetector CV Heuristic (Mock contour shape check or default None)
        badge_detector = BadgeDetector(badge_model=None)
        # Ensure it safely processes frames and returns customer or staff
        label, conf = badge_detector.identify_staff(frame, bbox)
        # Mock frame doesn't have exact name badge contour dimensions, so should return None
        self.assertIsNone(label)

        # 4. Test MultiModalStaffIdentifier Fusion
        fused_identifier = MultiModalStaffIdentifier(
            color_identifier=color_id,
            badge_detector=badge_detector,
            embedding_matcher=emb_matcher
        )
        
        # Physical check (Torso red apron is visible)
        label, conf = fused_identifier.identify_staff(frame, bbox)
        self.assertEqual(label, "staff_uniform")
        self.assertGreater(conf, 0.0)
        
        # Integrated check including Re-ID match
        label, conf = fused_identifier.identify_with_embedding(frame, bbox, query_mary)
        self.assertEqual(label, "emp_mary")
        self.assertGreater(conf, 0.0)

if __name__ == '__main__':
    unittest.main()
