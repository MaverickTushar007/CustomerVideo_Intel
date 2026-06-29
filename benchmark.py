from ingestion.frame_sampler import stream_frames
from ultralytics import YOLO
from tracking.position_tracker import PositionTracker
import sqlite3, time
from datetime import datetime, timezone, timedelta

VIDEOS = ['test_seated6.mp4', 'test_seated3.mkv', 'test_seated4.mkv']
MODELS = ['yolo11n.pt', 'yolov8m.pt']
VIDEO_START = datetime.now(timezone.utc)

def video_ts_to_iso(ts):
    return (VIDEO_START + timedelta(seconds=ts)).isoformat()

def run(model_path, video):
    detector = YOLO(model_path)
    tracker = PositionTracker(max_distance=150, max_missing_frames=150)
    
    db = sqlite3.connect('db/customer_intel.db')
    db.execute("DELETE FROM wait_metrics")
    db.execute("DELETE FROM persons")
    db.commit()

    total_detections = 0
    total_frames = 0
    t0 = time.time()

    for fid, ts, frame in stream_frames(video, fps_target=8):
        results = detector(frame, conf=0.25, classes=[0], verbose=False)
        bboxes = [box.xyxy[0].tolist() for box in results[0].boxes]
        total_detections += len(bboxes)
        total_frames += 1
        tracks = tracker.update(bboxes, fid, ts)
        for token, bbox, is_new in tracks:
            if is_new:
                db.execute(
                    "INSERT OR IGNORE INTO persons (token_id, first_seen, camera_id) VALUES (?,?,?)",
                    (token, video_ts_to_iso(ts), 'cam_01'))
        for token, entry_ts, exit_ts in tracker.get_exited(fid):
            wait = exit_ts - entry_ts
            if wait > 2:
                db.execute(
                    "INSERT INTO wait_metrics (token_id, entry_time, exit_time, wait_seconds, abandoned, date) VALUES (?,?,?,?,?,?)",
                    (token, video_ts_to_iso(entry_ts), video_ts_to_iso(exit_ts), round(wait,2), 1, VIDEO_START.strftime('%Y-%m-%d')))
                db.execute("UPDATE persons SET last_seen=? WHERE token_id=?", (video_ts_to_iso(exit_ts), token))

    for token, entry_ts, exit_ts in tracker.flush_all():
        wait = exit_ts - entry_ts
        if wait > 2:
            db.execute(
                "INSERT INTO wait_metrics (token_id, entry_time, exit_time, wait_seconds, abandoned, date) VALUES (?,?,?,?,?,?)",
                (token, video_ts_to_iso(entry_ts), video_ts_to_iso(exit_ts), round(wait,2), 1, VIDEO_START.strftime('%Y-%m-%d')))
            db.execute("UPDATE persons SET last_seen=? WHERE token_id=? AND last_seen IS NULL", (video_ts_to_iso(exit_ts), token))
    db.commit()

    elapsed = time.time() - t0

    tokens = db.execute("SELECT COUNT(DISTINCT token_id) FROM persons").fetchone()[0]
    avg_dwell = db.execute("SELECT ROUND(AVG(wait_seconds),1) FROM wait_metrics WHERE wait_seconds > 2").fetchone()[0]
    max_dwell = db.execute("SELECT ROUND(MAX(wait_seconds),1) FROM wait_metrics WHERE wait_seconds > 2").fetchone()[0]
    video_dur = db.execute("""
        SELECT CAST(strftime('%s', MAX(last_seen)) AS INTEGER) -
               CAST(strftime('%s', MIN(first_seen)) AS INTEGER)
        FROM persons
    """).fetchone()[0] or 1
    fragmentation = round(tokens / max(1, video_dur / 5), 2)
    db.close()

    return {
        'model': model_path,
        'video': video,
        'tokens': tokens,
        'avg_dwell': avg_dwell,
        'max_dwell': max_dwell,
        'fragmentation_ratio': fragmentation,
        'fps': round(total_frames / elapsed, 1),
        'elapsed': round(elapsed, 1)
    }

print(f"{'Model':<14} {'Video':<22} {'Tokens':<8} {'AvgDwell':<10} {'MaxDwell':<10} {'Frag':<6} {'FPS'}")
print('-'*85)
for model in MODELS:
    for video in VIDEOS:
        r = run(model, video)
        print(f"{r['model']:<14} {r['video']:<22} {str(r['tokens']):<8} {str(r['avg_dwell']):<10} {str(r['max_dwell']):<10} {str(r['fragmentation_ratio']):<6} {r['fps']}")
    print()
