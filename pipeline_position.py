from ingestion.frame_sampler import stream_frames
from ultralytics import YOLO
from tracking.position_tracker import PositionTracker
import cv2
import sqlite3
import os
from restaurant_analytics.staff_identifier import MultiModalStaffIdentifier, UniformColorIdentifier, BadgeDetector
from datetime import datetime, timezone, timedelta

detector = YOLO('yolo11n.pt')

CONF_THRESHOLD = 0.25
VIDEO_START = datetime.now(timezone.utc)

def video_ts_to_iso(ts):
    return (VIDEO_START + timedelta(seconds=ts)).isoformat()

token_colors = {}

SERVICE_ZONE = [(360, 180), (482, 180), (482, 360), (360, 360)]

def in_service_zone(centroid):
    x, y = centroid
    return 360 <= x <= 482 and 180 <= y <= 360

# Ensure the database directory exists and use an absolute path
db_dir = os.path.join(os.path.dirname(__file__), 'db')
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, 'customer_intel.db')
DB = sqlite3.connect(db_path, check_same_thread=False)
# Clear previous data
DB.execute("DELETE FROM wait_metrics")
DB.execute("DELETE FROM persons")
DB.commit()

def get_color(token_id):
    if token_id not in token_colors:
        import numpy as np
        np.random.seed(hash(token_id) % (2**32))
        token_colors[token_id] = tuple(int(x) for x in np.random.randint(80, 255, 3))
    return token_colors[token_id]

def log_person(token, entry_ts, camera_id, is_staff, staff_id):
    """Log a person entry into the database with explicit staff status.

    Parameters:
        token (str): Unique identifier for the person.
        entry_ts (float): Timestamp of entry in video seconds.
        camera_id (str): Identifier of the camera.
        is_staff (int): 1 if staff member, 0 otherwise.
        staff_id (str or None): Identifier for staff member if applicable.
    """
    DB.execute(
        "INSERT OR IGNORE INTO persons (token_id, first_seen, camera_id, is_staff, staff_id) VALUES (?,?,?,?,?)",
        (token, video_ts_to_iso(entry_ts), camera_id, is_staff, staff_id)
    )
    DB.commit()

def log_exit(token, entry_ts, exit_ts, served_tokens, service_times):
    wait = exit_ts - entry_ts
    if wait <= 2:
        return
    
    # Check if staff member
    is_staff_row = DB.execute("SELECT is_staff FROM persons WHERE token_id=?", (token,)).fetchone()
    is_staff = is_staff_row[0] if is_staff_row else 0
    
    date = VIDEO_START.strftime('%Y-%m-%d')
    abandoned = 0 if token in served_tokens or is_staff else 1
    
    # Calculate time to service for customer
    time_to_service = None
    if token in service_times and not is_staff:
        time_to_service = round(service_times[token] - entry_ts, 2)
        
    DB.execute(
        "INSERT INTO wait_metrics (token_id, entry_time, exit_time, wait_seconds, time_to_service, abandoned, date) VALUES (?,?,?,?,?,?,?)",
        (token, video_ts_to_iso(entry_ts), video_ts_to_iso(exit_ts), round(wait, 2), time_to_service, abandoned, date))
    DB.execute(
        "UPDATE persons SET last_seen=?, abandoned=? WHERE token_id=?",
        (video_ts_to_iso(exit_ts), abandoned, token))
    DB.commit()
    print(f"  [DB] {token} exited — dwell: {wait:.1f}s")

VIDEOS = [('test_seated4.mkv', 'cam_test')]

# Initialize multimodal staff identifier with multiple uniform color ranges
# Each covers a common restaurant staff uniform palette
# NOTE: Removed broad black range — it false-positives on dark customer clothing/shadows
color_identifiers = [
    UniformColorIdentifier(lower_hsv=(0, 120, 100), upper_hsv=(10, 255, 255), pixel_ratio_threshold=0.25),   # Red / maroon aprons
    UniformColorIdentifier(lower_hsv=(170, 120, 100), upper_hsv=(180, 255, 255), pixel_ratio_threshold=0.25), # Red wrap-around
    UniformColorIdentifier(lower_hsv=(100, 100, 60), upper_hsv=(125, 255, 200), pixel_ratio_threshold=0.25),  # Dark blue / navy
]

class MultiRangeUniformIdentifier:
    """Wraps multiple UniformColorIdentifier ranges into one."""
    def __init__(self, identifiers):
        self.identifiers = identifiers
    def identify_staff(self, frame, bbox):
        best_label, best_conf = None, 0.0
        for ident in self.identifiers:
            label, conf = ident.identify_staff(frame, bbox)
            if label and conf > best_conf:
                best_label, best_conf = label, conf
        return best_label, best_conf

identifier = MultiModalStaffIdentifier(
    color_identifier=MultiRangeUniformIdentifier(color_identifiers),
    badge_detector=BadgeDetector()
)

# Minimum confidence from the multimodal identifier to classify as staff
STAFF_CONFIDENCE_THRESHOLD = 0.30

# Visual constants for bounding box colors
STAFF_COLOR = (0, 255, 0)      # Green for staff
CUSTOMER_COLOR_WAIT = (0, 200, 255)  # Orange for waiting customers
CUSTOMER_COLOR_SERVED = (255, 200, 0)  # Cyan for served customers
ABANDON_COLOR = (0, 0, 255)    # Red for abandoned

PROXIMITY_THRESHOLD = 120  # Pixel distance for staff-customer "attend" event

print("Pipeline running — press Q to quit")

for _vid, _cam in VIDEOS:
    print(f"\n--- Processing {_cam} ---")
    tracker = PositionTracker(max_distance=150, max_missing_frames=150)
    served_tokens = set()
    token_is_staff = {}
    service_times = {}

    for fid, ts, frame in stream_frames(_vid, fps_target=8):
        results = detector(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)
        bboxes = [box.xyxy[0].tolist() for box in results[0].boxes]

        tracks = tracker.update(bboxes, fid, ts)

        # Build centroid lookup for proximity-based attend detection
        centroids = {}  # token -> (cx, cy)
        staff_centroids = {}  # token -> (cx, cy) for staff only

        for token, bbox, is_new in tracks:
            cx = (bbox[0]+bbox[2])/2
            cy = (bbox[1]+bbox[3])/2
            centroids[token] = (cx, cy)

            # Determine staff status (cache per token)
            if token not in token_is_staff:
                label, confidence = identifier.identify_staff(frame, bbox)
                is_staff = 1 if (label is not None and confidence >= STAFF_CONFIDENCE_THRESHOLD) else 0
                staff_id = f"emp_{token[:4]}" if is_staff else None
                token_is_staff[token] = is_staff
                # Log person entry with staff flag
                log_person(token, ts, _cam, is_staff, staff_id)
                conf_str = f" conf={confidence:.2f}" if label else ""
                print(f"[{ts:.1f}s] NEW person: {token} ({'STAFF' if is_staff else 'customer'}{conf_str})")
            else:
                is_staff = token_is_staff[token]

            if is_staff:
                staff_centroids[token] = (cx, cy)

            # Service zone handling
            if in_service_zone((cx, cy)):
                served_tokens.add(token)
                if not is_staff:
                    service_times.setdefault(token, ts)

        # Proximity-based attend: if any staff centroid is close to a customer, mark as served
        for token, (cx, cy) in centroids.items():
            if token_is_staff.get(token):
                continue  # skip staff themselves
            for s_token, (sx, sy) in staff_centroids.items():
                dist = ((cx - sx)**2 + (cy - sy)**2) ** 0.5
                if dist < PROXIMITY_THRESHOLD:
                    served_tokens.add(token)
                    service_times.setdefault(token, ts)
                    break

        # Draw bounding boxes with role-based colors
        for token, bbox, is_new in tracks:
            is_staff = token_is_staff.get(token, 0)
            x1, y1, x2, y2 = [int(v) for v in bbox]

            if is_staff:
                color = STAFF_COLOR
                label_text = f"STAFF {token[:6]}"
            elif token in served_tokens:
                color = CUSTOMER_COLOR_SERVED
                label_text = f"SERVED {token[:6]}"
            else:
                color = CUSTOMER_COLOR_WAIT
                label_text = f"WAIT {token[:6]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, max(y1-18, 0)), (x1+tw+4, max(y1-2, 0)), color, -1)
            cv2.putText(frame, label_text, (x1+2, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        for token, entry_ts, exit_ts in tracker.get_exited(fid):
            log_exit(token, entry_ts, exit_ts, served_tokens, service_times)

        # Draw service zone
        cv2.rectangle(frame, (360,180), (482,360), (0,165,255), 2)
        cv2.putText(frame, 'SERVICE ZONE', (362,175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)

        # Stats overlay
        n_staff = sum(1 for v in token_is_staff.values() if v)
        n_cust = len(token_is_staff) - n_staff
        cv2.putText(frame, f"Tracked: {len(tracker.tracks)} | Staff: {n_staff} | Cust: {n_cust} | t={ts:.1f}s",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        try:
            cv2.imshow('Position Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            pass

    for token, entry_ts, exit_ts in tracker.flush_all():
        log_exit(token, entry_ts, exit_ts, served_tokens, service_times)
        DB.execute(
            "UPDATE persons SET last_seen=? WHERE token_id=? AND last_seen IS NULL",
            (video_ts_to_iso(exit_ts), token))
    DB.commit()
    print(f"--- Done {_cam} ---")

try:
    cv2.destroyAllWindows()
except Exception:
    pass
DB.close()
print("\nAll done.")
