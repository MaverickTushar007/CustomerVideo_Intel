from ingestion.frame_sampler import stream_frames
from ultralytics import YOLO
import torchreid
import torch
import numpy as np
import cv2
import uuid

detector = YOLO('yolo11n.pt')
embedder = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
embedder.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder.to(device)

# --- TUNED PARAMS ---
CONF_THRESHOLD  = 0.25   # was 0.4 — catches more people
REID_THRESHOLD  = 0.65   # was 0.75 — more lenient cross-angle matching
MIN_BOX_HEIGHT  = 40     # skip crops too small to embed reliably
FPS_TARGET      = 12     # was 8 — more frames = better tracking continuity

gallery = {}
token_colors = {}

def get_embedding(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    if h < MIN_BOX_HEIGHT:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (128, 256))
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    crop = np.transpose(crop, (2,0,1))
    tensor = torch.tensor(crop).unsqueeze(0).float().to(device)
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
        return best_token, False, best_score
    else:
        new_token = str(uuid.uuid4())[:8]
        gallery[new_token] = embedding
        return new_token, True, best_score

def get_color(token_id):
    if token_id not in token_colors:
        np.random.seed(hash(token_id) % (2**32))
        token_colors[token_id] = tuple(int(x) for x in np.random.randint(80, 255, 3))
    return token_colors[token_id]

new_this_run = 0
for fid, ts, frame in stream_frames('test_video.mp4', fps_target=FPS_TARGET):
    results = detector(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        bbox = box.xyxy[0].tolist()
        emb = get_embedding(frame, bbox)
        if emb is None:
            continue
        token, is_new, score = assign_token(emb)
        if is_new:
            new_this_run += 1
        color = get_color(token)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f'NEW {token}' if is_new else f'{token} {score:.2f}'
        cv2.putText(frame, label, (x1, max(y1-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # HUD
    cv2.putText(frame, f'Unique: {len(gallery)}  |  t={ts:.1f}s', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(frame, f'conf={CONF_THRESHOLD}  reid={REID_THRESHOLD}', (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    cv2.imshow('Customer Tracker v2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f'Total unique persons: {len(gallery)}')
print(f'Conf={CONF_THRESHOLD} | ReID threshold={REID_THRESHOLD} | FPS={FPS_TARGET}')
