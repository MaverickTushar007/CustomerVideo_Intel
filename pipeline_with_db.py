from ingestion.frame_sampler import stream_frames
from ultralytics import YOLO
import torchreid
import torch
import numpy as np
import cv2
import uuid
import sqlite3
from datetime import datetime, timezone, timedelta

detector = YOLO('yolo11m.pt')
embedder = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
embedder.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder.to(device)

CONF_THRESHOLD = 0.15
REID_THRESHOLD = 0.70
MIN_BOX_HEIGHT = 20
CAMERA_ID = 'cam_01'

# Anchor: treat video t=0 as current real time
VIDEO_START = datetime.now(timezone.utc)

def video_ts_to_iso(ts_seconds):
    """Convert video timestamp (float seconds) to ISO8601 string."""
    return (VIDEO_START + timedelta(seconds=ts_seconds)).isoformat()

gallery = {}
tracks = {}
token_colors = {}
last_seen_frame = {}
last_seen_ts = {}

DB = sqlite3.connect('db/customer_intel.db')

# Clear previous test data
DB.execute("DELETE FROM wait_metrics")
DB.execute("DELETE FROM persons")
DB.commit()

def get_embedding(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if (y2 - y1) < MIN_BOX_HEIGHT:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (128, 256))
    inp = crop.astype(np.float32) / 255.0
    inp = (inp - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    inp = np.transpose(inp, (2,0,1))
    tensor = torch.tensor(inp).unsqueeze(0).float().to(device)
    with torch.no_grad():
        feat = embedder(tensor)
    feat = feat.cpu().numpy().flatten()
    return feat / (np.linalg.norm(feat) + 1e-8)

def assign_token(embedding):
    best_token, best_score = None, 0.0
    for token_id, stored_emb in gallery.items():
        score = float(np.dot(embedding, stored_emb))
        if score > best_score:
            best_score = score
            best_token = token_id
    if best_score >= REID_THRESHOLD:
        gallery[best_token] = 0.9 * gallery[best_token] + 0.1 * embedding
        gallery[best_token] /= np.linalg.norm(gallery[best_token])
        return best_token, False
    else:
        new_token = str(uuid.uuid4())[:8]
        gallery[new_token] = embedding
        return new_token, True

def get_color(token_id):
    if token_id not in token_colors:
        np.random.seed(hash(token_id) % (2**32))
        token_colors[token_id] = tuple(int(x) for x in np.random.randint(80, 255, 3))
    return token_colors[token_id]

def register_person(token, entry_iso):
    cur = DB.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO persons (token_id, first_seen, camera_id) VALUES (?,?,?)",
        (token, entry_iso, CAMERA_ID)
    )
    DB.commit()

def log_exit(token, track, exit_ts):
    entry_iso = track['entry_iso']
    exit_iso = video_ts_to_iso(exit_ts)
    wait = exit_ts - track['entry_ts']
    date = VIDEO_START.strftime('%Y-%m-%d')
    cur = DB.cursor()
    cur.execute(
        "INSERT INTO wait_metrics (token_id, entry_time, exit_time, wait_seconds, abandoned, date) VALUES (?,?,?,?,?,?)",
        (token, entry_iso, exit_iso, round(wait, 2), 1, date)
    )
    cur.execute("UPDATE persons SET last_seen=?, abandoned=1 WHERE token_id=?", (exit_iso, token))
    DB.commit()
    print(f"  [DB] {token} exited — dwell: {wait:.1f}s (video time)")

print("Pipeline running — press Q to quit")
for fid, ts, frame in stream_frames('test_restaurant.mp4.webm', fps_target=12):
    results = detector(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)

    for box in results[0].boxes:
        emb = get_embedding(frame, box.xyxy[0].tolist())
        if emb is None:
            continue
        token, is_new = assign_token(emb)
        last_seen_frame[token] = fid
        last_seen_ts[token] = ts

        if is_new:
            entry_iso = video_ts_to_iso(ts)
            tracks[token] = {'entry_ts': ts, 'entry_iso': entry_iso}
            register_person(token, entry_iso)
            print(f"[{ts:.1f}s] NEW person: {token}")

        color = get_color(token)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{token} {ts:.1f}s", (x1, max(y1-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Exit anyone not seen for 30+ frames
    for token, last_fid in list(last_seen_frame.items()):
        if token in tracks and (fid - last_fid) > 80:  # 10s at 8fps for seated venues
            log_exit(token, tracks[token], last_seen_ts[token])
            del tracks[token]
            del last_seen_frame[token]
            del last_seen_ts[token]

    cv2.putText(frame, f"Unique: {len(gallery)} | t={ts:.1f}s",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow('Pipeline + DB', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Log remaining
for token, track in tracks.items():
    log_exit(token, track, last_seen_ts.get(token, 0))

cv2.destroyAllWindows()
DB.close()
print(f"\nDone. {len(gallery)} unique persons logged.")
