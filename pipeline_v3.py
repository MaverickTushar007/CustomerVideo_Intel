from ultralytics import YOLO
import cv2
import numpy as np
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta

detector = YOLO('yolo11x.pt')

CONF_THRESHOLD = 0.20
MIN_BOX_HEIGHT = 25
REID_THRESHOLD = 0.68
CAMERA_ID = 'cam_01'
VIDEO_PATH = 'test_video3.mp4'

VIDEO_START = datetime.now(timezone.utc)

def video_ts_to_iso(ts):
    return (VIDEO_START + timedelta(seconds=ts)).isoformat()

import torchreid, torch
embedder = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
embedder.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder.to(device)

DB = sqlite3.connect('db/customer_intel.db')
DB.execute("DELETE FROM wait_metrics")
DB.execute("DELETE FROM persons")
DB.commit()

gallery = {}
tracks = {}
last_seen_frame = {}
last_seen_ts = {}
token_colors = {}

def nms_boxes(boxes, iou_threshold=0.45):
    """Remove duplicate detections from tiling overlap."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    scores = boxes[:,4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return boxes[keep].tolist()

def tiled_detect(frame):
    """Run detection on full frame + 4 quadrants, merge with NMS."""
    h, w = frame.shape[:2]
    all_dets = []
    tiles = [
        (frame, 0, 0),
        (frame[0:h//2, 0:w//2], 0, 0),
        (frame[0:h//2, w//2:], w//2, 0),
        (frame[h//2:, 0:w//2], 0, h//2),
        (frame[h//2:, w//2:], w//2, h//2),
    ]
    for tile, ox, oy in tiles:
        results = detector(tile, conf=CONF_THRESHOLD, classes=[0], verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            all_dets.append([x1+ox, y1+oy, x2+ox, y2+oy, conf])
    return nms_boxes(all_dets, iou_threshold=0.45)

def get_embedding(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
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
    for tid, stored in gallery.items():
        score = float(np.dot(embedding, stored))
        if score > best_score:
            best_score = score
            best_token = tid
    if best_score >= REID_THRESHOLD:
        gallery[best_token] = 0.9*gallery[best_token] + 0.1*embedding
        gallery[best_token] /= np.linalg.norm(gallery[best_token])
        return best_token, False
    else:
        new_token = str(uuid.uuid4())[:8]
        gallery[new_token] = embedding
        return new_token, True

def get_color(tid):
    if tid not in token_colors:
        np.random.seed(hash(tid) % (2**32))
        token_colors[tid] = tuple(int(x) for x in np.random.randint(80, 255, 3))
    return token_colors[tid]

def register_person(token, entry_iso):
    cur = DB.cursor()
    cur.execute("INSERT OR IGNORE INTO persons (token_id, first_seen, camera_id) VALUES (?,?,?)",
                (token, entry_iso, CAMERA_ID))
    DB.commit()

def log_exit(token, track, exit_ts):
    wait = exit_ts - track['entry_ts']
    if wait < 1.0:
        return
    exit_iso = video_ts_to_iso(exit_ts)
    date = VIDEO_START.strftime('%Y-%m-%d')
    cur = DB.cursor()
    cur.execute(
        "INSERT INTO wait_metrics (token_id, entry_time, exit_time, wait_seconds, abandoned, date) VALUES (?,?,?,?,?,?)",
        (token, track['entry_iso'], exit_iso, round(wait,2), 1, date))
    cur.execute("UPDATE persons SET last_seen=?, abandoned=1 WHERE token_id=?",
                (exit_iso, token))
    DB.commit()
    print(f"  [DB] {token} exited — dwell: {wait:.1f}s")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0
print("Pipeline v3 (YOLO11x + tiled NMS) — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if frame_id % 15 != 0:
        frame_id += 1
        continue
    boxes = tiled_detect(frame)

    for bbox in boxes:
        emb = get_embedding(frame, bbox)
        if emb is None:
            continue
        token, is_new = assign_token(emb)
        last_seen_frame[token] = frame_id
        last_seen_ts[token] = ts

        if is_new:
            entry_iso = video_ts_to_iso(ts)
            tracks[token] = {'entry_ts': ts, 'entry_iso': entry_iso}
            register_person(token, entry_iso)
            print(f"[{ts:.1f}s] NEW: {token}")

        color = get_color(token)
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, token, (x1, max(y1-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Exit tokens not seen for 45+ frames
    for token, last_fid in list(last_seen_frame.items()):
        if token in tracks and (frame_id - last_fid) > 45:
            log_exit(token, tracks[token], last_seen_ts[token])
            del tracks[token]
            del last_seen_frame[token]
            del last_seen_ts[token]

    cv2.putText(frame, f"Unique: {len(gallery)} | Active: {len(tracks)} | t={ts:.1f}s",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow('Pipeline v3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_id += 1

for token, track in tracks.items():
    log_exit(token, track, last_seen_ts.get(token, track['entry_ts']))

cap.release()
cv2.destroyAllWindows()
DB.close()
print(f"\nDone. {len(gallery)} unique persons logged.")
