from boxmot.trackers import ByteTrack
from ultralytics import YOLO
import cv2
import numpy as np
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Models
detector = YOLO('yolo11m.pt')
tracker = ByteTrack()

CONF_THRESHOLD = 0.15
MIN_BOX_HEIGHT = 20
CAMERA_ID = 'cam_01'
VIDEO_PATH = 'test_video_airport.mp4'

VIDEO_START = datetime.now(timezone.utc)

def video_ts_to_iso(ts):
    return (VIDEO_START + timedelta(seconds=ts)).isoformat()

# DB setup
DB = sqlite3.connect('db/customer_intel.db')
DB.execute("DELETE FROM wait_metrics")
DB.execute("DELETE FROM persons")
DB.commit()

tracks = {}        # track_id -> {entry_ts, entry_iso}
last_seen_ts = {}  # track_id -> last video ts
last_seen_frame = {} # track_id -> last frame id
token_map = {}     # track_id -> short token string
token_colors = {}  # token -> color

def get_color(tid):
    if tid not in token_colors:
        np.random.seed(int(tid) % (2**32))
        token_colors[tid] = tuple(int(x) for x in np.random.randint(80, 255, 3))
    return token_colors[tid]

def register_person(tid, entry_iso):
    token = str(uuid.uuid4())[:8]
    token_map[tid] = token
    cur = DB.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO persons (token_id, first_seen, camera_id) VALUES (?,?,?)",
        (token, entry_iso, CAMERA_ID))
    DB.commit()
    return token

def log_exit(tid, exit_ts):
    if tid not in tracks:
        return
    track = tracks[tid]
    token = token_map.get(tid, str(tid))
    wait = exit_ts - track['entry_ts']
    if wait < 0.5:
        return
    exit_iso = video_ts_to_iso(exit_ts)
    date = VIDEO_START.strftime('%Y-%m-%d')
    cur = DB.cursor()
    cur.execute(
        "INSERT INTO wait_metrics (token_id, entry_time, exit_time, wait_seconds, abandoned, date) VALUES (?,?,?,?,?,?)",
        (token, track['entry_iso'], exit_iso, round(wait, 2), 1, date))
    cur.execute(
        "UPDATE persons SET last_seen=?, abandoned=1 WHERE token_id=?",
        (exit_iso, token))
    DB.commit()
    print(f"  [DB] {token} exited — dwell: {wait:.1f}s")

cap = cv2.VideoCapture(VIDEO_PATH)
native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_id = 0

print("ByteTrack pipeline running — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Detect
    results = detector(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)
    dets = results[0].boxes.data.cpu().numpy() if len(results[0].boxes) else np.empty((0, 6))

    # Track
    tracked = tracker.update(dets, frame)
    # tracked cols: x1, y1, x2, y2, track_id, conf, class, ?

    active_ids = set()
    for t in tracked:
        x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
        if (y2 - y1) < MIN_BOX_HEIGHT:
            continue

        active_ids.add(tid)
        last_seen_frame[tid] = frame_id
        last_seen_ts[tid] = ts

        if tid not in tracks:
            entry_iso = video_ts_to_iso(ts)
            tracks[tid] = {'entry_ts': ts, 'entry_iso': entry_iso}
            token = register_person(tid, entry_iso)
            print(f"[{ts:.1f}s] NEW person: {token} (track {tid})")

        color = get_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{token_map.get(tid, str(tid))} {ts:.1f}s",
                    (x1, max(y1-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Exit anyone not seen for 60+ frames
    for tid, last_fid in list(last_seen_frame.items()):
        if tid in tracks and (frame_id - last_fid) > 60:
            log_exit(tid, last_seen_ts[tid])
            del tracks[tid]
            del last_seen_frame[tid]
            del last_seen_ts[tid]

    cv2.putText(frame, f"Unique: {len(token_map)} | Active: {len(tracks)} | t={ts:.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('ByteTrack Pipeline', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Log remaining active tracks
for tid, track in tracks.items():
    log_exit(tid, last_seen_ts.get(tid, track['entry_ts']))

cap.release()
cv2.destroyAllWindows()
DB.close()
print(f"\nDone. {len(token_map)} unique persons logged.")
