import os, uuid, threading, time
import cv2
import sqlite3
from datetime import datetime, timezone, timedelta
from ultralytics import YOLO
from tracking.position_tracker import PositionTracker

# Absolute paths — works on Railway regardless of working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "customer_intel.db")

jobs = {}

detector = None

def load_models():
    global detector
    if detector is None:
        detector = YOLO('yolo_staff_customer.pt')

def in_service_zone(centroid):
    x, y = centroid
    return 360 <= x <= 482 and 180 <= y <= 360

def process_video(job_id, video_path, venue_id="default"):
    try:
        jobs[job_id] = {"status": "running", "progress": 0,
                        "stage": "Loading models...", "result": None,
                        "venue_id": venue_id}
        load_models()

        jobs[job_id]["stage"] = "Detecting people..."
        jobs[job_id]["progress"] = 10

        VIDEO_START = datetime.now(timezone.utc)
        tracker = PositionTracker(max_distance=150, max_missing_frames=150)

        def video_ts_to_iso(ts):
            return (VIDEO_START + timedelta(seconds=ts)).isoformat()

        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        skip = max(1, int(native_fps / 8))
        frame_id = 0

        DB = sqlite3.connect(DB_PATH)

        token_entry = {}
        token_class_votes = {}
        token_is_staff = {}
        resolved_staff_tokens = set()
        served_tokens = set()
        service_times = {}

        PROXIMITY_THRESHOLD = 120

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % skip == 0:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                progress = 10 + int((frame_id / max(total_frames, 1)) * 75)
                jobs[job_id]["progress"] = progress
                jobs[job_id]["stage"] = "Detecting people..." if progress < 40 else "Tracking visitors..."

                results = detector(frame, conf=0.25, verbose=False)
                bboxes = []
                box_classes = []
                box_confidences = []
                for box in results[0].boxes:
                    bboxes.append(box.xyxy[0].tolist())
                    box_classes.append(int(box.cls[0]))
                    box_confidences.append(float(box.conf[0]))

                tracks = tracker.update(bboxes, frame_id, ts)

                # Centroid positions for proximity attendance checks
                centroids = {}
                staff_centroids = {}

                for i, (token, bbox, is_new) in enumerate(tracks):
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    centroids[token] = (cx, cy)

                    # Safeguard array boundaries
                    cls = box_classes[i] if i < len(box_classes) else 0
                    conf = box_confidences[i] if i < len(box_confidences) else 0.5

                    token_class_votes.setdefault(token, []).append(cls)
                    votes = token_class_votes[token]
                    is_staff = 1 if (votes.count(1) > votes.count(0)) else 0
                    token_is_staff[token] = is_staff

                    if is_new:
                        token_entry[token] = ts
                        DB.execute(
                            "INSERT OR IGNORE INTO temporal_sessions (session_id, camera_id, start_time) VALUES (?,?,?)",
                            (token, 'cam_01', video_ts_to_iso(ts)))
                        DB.commit()

                    if is_staff and token not in resolved_staff_tokens:
                        resolved_staff_tokens.add(token)
                        staff_id = f"staff_{token[:4]}"
                        DB.execute(
                            "INSERT INTO staff_resolutions (session_id, staff_id, confidence, resolution_method, resolved_at) VALUES (?,?,?,?,?)",
                            (token, staff_id, round(conf, 3), "yolo_custom", datetime.now(timezone.utc).isoformat())
                        )
                        DB.commit()

                    if is_staff:
                        staff_centroids[token] = (cx, cy)

                    # Service zone handling
                    if in_service_zone((cx, cy)):
                        served_tokens.add(token)
                        if not is_staff:
                            service_times.setdefault(token, ts)

                # Proximity attend check
                for token, (cx, cy) in centroids.items():
                    if token_is_staff.get(token):
                        continue
                    for s_token, (sx, sy) in staff_centroids.items():
                        dist = ((cx - sx)**2 + (cy - sy)**2) ** 0.5
                        if dist < PROXIMITY_THRESHOLD:
                            served_tokens.add(token)
                            service_times.setdefault(token, ts)
                            break

                for token, entry_ts, exit_ts in tracker.get_exited(frame_id):
                    duration = exit_ts - entry_ts
                    if duration > 2:
                        end_iso = video_ts_to_iso(exit_ts)
                        DB.execute("UPDATE temporal_sessions SET end_time=?, duration_seconds=? WHERE session_id=?", 
                                   (end_iso, round(duration, 2), token))
                        
                        is_staff = token_is_staff.get(token, 0)
                        if is_staff:
                            event_type = "staff_shift"
                            val = round(duration, 2)
                        elif token in served_tokens:
                            event_type = "served"
                            val = round(service_times[token] - entry_ts, 2) if token in service_times else round(duration, 2)
                        else:
                            event_type = "abandoned"
                            val = round(duration, 2)

                        DB.execute("INSERT INTO business_events (session_id, event_type, timestamp, value) VALUES (?,?,?,?)",
                                   (token, event_type, end_iso, val))
                        DB.commit()

            frame_id += 1

        for token, entry_ts, exit_ts in tracker.flush_all():
            duration = exit_ts - entry_ts
            if duration > 2:
                end_iso = video_ts_to_iso(exit_ts)
                DB.execute("UPDATE temporal_sessions SET end_time=?, duration_seconds=? WHERE session_id=?", 
                           (end_iso, round(duration, 2), token))

                is_staff = token_is_staff.get(token, 0)
                if is_staff:
                    event_type = "staff_shift"
                    val = round(duration, 2)
                elif token in served_tokens:
                    event_type = "served"
                    val = round(service_times[token] - entry_ts, 2) if token in service_times else round(duration, 2)
                else:
                    event_type = "abandoned"
                    val = round(duration, 2)

                DB.execute("INSERT INTO business_events (session_id, event_type, timestamp, value) VALUES (?,?,?,?)",
                           (token, event_type, end_iso, val))
                DB.commit()

        cap.release()
        DB.close()
        os.remove(video_path)

        # ── Self-learning: update venue memory from this run ──────────────
        jobs[job_id]["stage"] = "Learning from this run..."
        try:
            from db.memory_updater import run_update
            run_update(venue_id=venue_id)
        except Exception as mem_err:
            print(f"  [MEMORY] Warning: could not update venue memory: {mem_err}")

        jobs[job_id]["stage"] = "Complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = "done"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["stage"] = f"Error: {str(e)}"

def start_job(video_path, venue_id="default"):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0, "stage": "Queued",
                    "result": None, "venue_id": venue_id}
    t = threading.Thread(target=process_video, args=(job_id, video_path, venue_id))
    t.daemon = True
    t.start()
    return job_id
