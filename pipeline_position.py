from ingestion.frame_sampler import stream_frames
from ultralytics import YOLO
from tracking.position_tracker import PositionTracker
import cv2
import sqlite3
import os
from restaurant_analytics.staff_identifier import MultiModalStaffIdentifier, UniformColorIdentifier, BadgeDetector
from datetime import datetime, timezone, timedelta

detector = YOLO('yolo_staff_customer.pt')

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
# Clear previous run data from all decoupled tables
for _tbl in ("business_events", "staff_resolutions", "temporal_sessions", "raw_observations"):
    DB.execute(f"DELETE FROM {_tbl}")
DB.commit()

def get_color(token_id):
    if token_id not in token_colors:
        import numpy as np
        np.random.seed(hash(token_id) % (2**32))
        token_colors[token_id] = tuple(int(x) for x in np.random.randint(80, 255, 3))
    return token_colors[token_id]

# ── DB write helpers (new decoupled schema) ──────────────────────────────────

def log_session_start(session_id: str, camera_id: str, start_ts: float):
    """Open a temporal session row for this tracker lifecycle."""
    DB.execute(
        "INSERT OR IGNORE INTO temporal_sessions (session_id, camera_id, start_time) VALUES (?,?,?)",
        (session_id, camera_id, video_ts_to_iso(start_ts))
    )
    DB.commit()


def log_staff_resolution(
    session_id: str,
    staff_id: str | None,
    confidence: float,
    method: str
):
    """Write a staff identity resolution — only called when confidence >= threshold."""
    DB.execute(
        """INSERT INTO staff_resolutions
               (session_id, staff_id, confidence, resolution_method, resolved_at)
           VALUES (?,?,?,?,?)""",
        (session_id, staff_id, round(confidence, 3), method,
         datetime.now(timezone.utc).isoformat())
    )
    DB.commit()


def log_session_end(
    session_id: str,
    entry_ts: float,
    exit_ts: float,
    served_tokens: set,
    service_times: dict,
    is_staff: int
):
    """Close a temporal session and emit derived business events."""
    duration = exit_ts - entry_ts
    if duration <= 2:
        return

    end_iso = video_ts_to_iso(exit_ts)

    # 1. Close the temporal session
    DB.execute(
        "UPDATE temporal_sessions SET end_time=?, duration_seconds=? WHERE session_id=?",
        (end_iso, round(duration, 2), session_id)
    )

    # 2. Emit business event — staff sessions get a simple 'staff_shift' marker,
    #    customer sessions get 'served' or 'abandoned'
    if is_staff:
        event_type = "staff_shift"
        value = round(duration, 2)
    elif session_id in served_tokens:
        event_type = "served"
        value = round(
            service_times[session_id] - entry_ts, 2
        ) if session_id in service_times else round(duration, 2)
    else:
        event_type = "abandoned"
        value = round(duration, 2)

    DB.execute(
        """INSERT INTO business_events
               (session_id, event_type, timestamp, value)
           VALUES (?,?,?,?)""",
        (session_id, event_type, end_iso, value)
    )
    DB.commit()
    print(f"  [DB] {session_id[:8]} exited — {event_type} dwell:{duration:.1f}s")

VIDEOS = [('test_seated3.mkv', 'cam_test')]

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
STAFF_CONFIDENCE_THRESHOLD = 0.40

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
    token_class_votes = {}
    resolved_staff_tokens = set()

    writer = None
    for fid, ts, frame in stream_frames(_vid, fps_target=8):
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter('output_test_seated3.mp4', fourcc, 8, (w, h))
        results = detector(frame, conf=CONF_THRESHOLD, verbose=False)
        bboxes = []
        box_classes = []
        box_confidences = []
        for box in results[0].boxes:
            bboxes.append(box.xyxy[0].tolist())
            box_classes.append(int(box.cls[0]))
            box_confidences.append(float(box.conf[0]))

        tracks = tracker.update(bboxes, fid, ts)

        # Build centroid lookup for proximity-based attend detection
        centroids = {}  # token -> (cx, cy)
        staff_centroids = {}  # token -> (cx, cy) for staff only

        for i, (token, bbox, is_new) in enumerate(tracks):
            cx = (bbox[0]+bbox[2])/2
            cy = (bbox[1]+bbox[3])/2
            centroids[token] = (cx, cy)

            cls = box_classes[i]
            conf = box_confidences[i]

            # Update votes and dynamic role cache
            token_class_votes.setdefault(token, []).append(cls)
            votes = token_class_votes[token]
            is_staff = 1 if (votes.count(1) > votes.count(0)) else 0
            token_is_staff[token] = is_staff

            if is_new:
                log_session_start(token, _cam, ts)
                print(f"[{ts:.1f}s] NEW person: {token}")

            # If classification settles as staff, write resolution record
            if is_staff and token not in resolved_staff_tokens:
                resolved_staff_tokens.add(token)
                staff_id = f"staff_{token[:4]}"
                log_staff_resolution(
                    session_id=token,
                    staff_id=staff_id,
                    confidence=conf,
                    method="yolo_custom"
                )
                print(f"[{ts:.1f}s] Resolved {token} as STAFF (conf={conf:.2f})")

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
            log_session_end(token, entry_ts, exit_ts, served_tokens, service_times,
                            is_staff=token_is_staff.get(token, 0))

        # Draw service zone
        cv2.rectangle(frame, (360,180), (482,360), (0,165,255), 2)
        cv2.putText(frame, 'SERVICE ZONE', (362,175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)

        # Stats overlay
        n_staff = sum(1 for v in token_is_staff.values() if v)
        n_cust = len(token_is_staff) - n_staff
        cv2.putText(frame, f"Tracked: {len(tracker.tracks)} | Staff: {n_staff} | Cust: {n_cust} | t={ts:.1f}s",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        if writer is not None:
            writer.write(frame)

        try:
            cv2.imshow('Position Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            pass

    if writer is not None:
        writer.release()

    for token, entry_ts, exit_ts in tracker.flush_all():
        log_session_end(token, entry_ts, exit_ts, served_tokens, service_times,
                        is_staff=token_is_staff.get(token, 0))
    DB.commit()
    print(f"--- Done {_cam} ---")

try:
    cv2.destroyAllWindows()
except Exception:
    pass
DB.close()
print("\nAll done.")
