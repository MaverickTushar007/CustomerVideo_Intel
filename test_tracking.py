from ingestion.frame_sampler import stream_frames
from ultralytics import YOLO
import torchreid
import torch
import numpy as np
import cv2
import uuid

# --- Models ---
detector = YOLO('yolo11n.pt')
embedder = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
embedder.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder.to(device)

REID_THRESHOLD = 0.75
gallery = {}  # token_id -> embedding

def get_embedding(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
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
        # Update with running average
        gallery[best_token] = 0.9 * gallery[best_token] + 0.1 * embedding
        gallery[best_token] /= np.linalg.norm(gallery[best_token])
        return best_token, False, best_score
    else:
        new_token = str(uuid.uuid4())[:8]
        gallery[new_token] = embedding
        return new_token, True, best_score

# --- Colors per token for visualization ---
token_colors = {}
def get_color(token_id):
    if token_id not in token_colors:
        np.random.seed(hash(token_id) % (2**32))
        token_colors[token_id] = tuple(int(x) for x in np.random.randint(50, 255, 3))
    return token_colors[token_id]

print('Starting tracking — press Q to quit')
for fid, ts, frame in stream_frames('test_video.mp4', fps_target=8):
    results = detector(frame, conf=0.4, classes=[0], verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        bbox = box.xyxy[0].tolist()
        emb = get_embedding(frame, bbox)
        if emb is None:
            continue
        token, is_new, score = assign_token(emb)
        label = f'NEW:{token}' if is_new else f'{token} ({score:.2f})'
        color = get_color(token)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.putText(frame, f'Unique persons: {len(gallery)}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f't={ts:.1f}s', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow('Customer Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f'\nTracking complete. Total unique persons identified: {len(gallery)}')
for token in list(gallery.keys()):
    print(f'  Person {token}')
