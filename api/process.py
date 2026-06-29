import os, uuid, threading, time
import cv2
import sqlite3
from datetime import datetime, timezone, timedelta
from ultralytics import YOLO
from tracking.position_tracker import PositionTracker

jobs = {}

detector = None

def load_models():
    global detector
    if detector is None:
        detector = YOLO('yolo11n.pt')

def process_video(job_id, video_path, venue_id="default"):
    try:
        jobs[job_id] = {"status": "running", "progress": 0,
                        "stage": "Loading models...", "result": None,
                        "venue_id": venue_id}
        load_models()

        jobs[job_id]["stage"] = "Detecting people..."
        jobs[job_id]["progress"] = 10

        VIDEO_START = datetime.now(timezone.utc)
        tracker = PositionTracker(max_distance=120, max_missing_frames=100)

        def video_ts_to_iso(ts):
            return (VIDEO_START + timedelta(seconds=ts)).isoformat()

        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        skip = max(1, int(native_fps / 8))
        frame_id = 0

        DB = sqlite3.connect('db/customer_intel.db')
        DB.execute("DELETE FROM wait_metrics")
        DB.execute("DELETE FROM persons")
        DB.commit()

        token_entry = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % skip == 0:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                progress = 10 + int((frame_id / max(total_frames, 1)) * 75)
                jobs[job_id]["progress"] = progress
                jobs[job_id]["stage"] = "Detecting people..." if progress < 40 else "Tracking visitors..."

                results = detector(frame, conf=0.25, classes=[0], verbose=False)
                bboxes = [box.xyxy[0].tolist() for box in results[0].boxes]
                tracks = tracker.update(bboxes, frame_id, ts)

                for token, bbox, is_new in tracks:
                    if is_new:
                        token_entry[token] = ts
                        DB.execute(
                            "INSERT OR IGNORE INTO persons (token_id, first_seen, camera_id) VALUES (?,?,?)",
                            (token, video_ts_to_iso(ts), 'cam_01'))
                        DB.commit()

                for token, entry_ts, exit_ts in tracker.get_exited(frame_id):
                    wait = exit_ts - entry_ts
                    if wait > 2:
                        date = VIDEO_START.strftime('%Y-%m-%d')
                        DB.execute(
                            "INSERT INTO wait_metrics (token_id, entry_time, exit_time, wait_seconds, abandoned, date) VALUES (?,?,?,?,?,?)",
                            (token, video_ts_to_iso(entry_ts), video_ts_to_iso(exit_ts), round(wait, 2), 1, date))
                        DB.execute(
                            "UPDATE persons SET last_seen=?, abandoned=1 WHERE token_id=?",
                            (video_ts_to_iso(exit_ts), token))
                        DB.commit()

            frame_id += 1

        for token, entry_ts, exit_ts in tracker.flush_all():
            wait = exit_ts - entry_ts
            if wait > 2:
                date = VIDEO_START.strftime('%Y-%m-%d')
                DB.execute(
                    "INSERT INTO wait_metrics (token_id, entry_time, exit_time, wait_seconds, abandoned, date) VALUES (?,?,?,?,?,?)",
                    (token, video_ts_to_iso(entry_ts), video_ts_to_iso(exit_ts), round(wait, 2), 1, date))
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
