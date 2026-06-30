from ingestion.frame_sampler import stream_frames
from ultralytics import YOLO
import torchreid
import torch
import numpy as np
import cv2
import uuid
import os

detector = YOLO('yolo11n.pt')
embedder = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
embedder.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder.to(device)

CONF_THRESHOLD = 0.25
REID_THRESHOLD = 0.65
MIN_BOX_HEIGHT = 40

gallery = {}
token_crops = {}  # token_id -> list of crop images (max 3)

os.makedirs('token_crops', exist_ok=True)

def get_embedding(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if (y2 - y1) < MIN_BOX_HEIGHT:
        return None, None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    resized = cv2.resize(crop, (128, 256))
    inp = resized.astype(np.float32) / 255.0
    inp = (inp - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    inp = np.transpose(inp, (2,0,1))
    tensor = torch.tensor(inp).unsqueeze(0).float().to(device)
    with torch.no_grad():
        feat = embedder(tensor)
    feat = feat.cpu().numpy().flatten()
    return feat / (np.linalg.norm(feat) + 1e-8), resized

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

for fid, ts, frame in stream_frames('test_video.mp4', fps_target=12):
    results = detector(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)
    for box in results[0].boxes:
        emb, crop = get_embedding(frame, box.xyxy[0].tolist())
        if emb is None:
            continue
        token, is_new = assign_token(emb)
        if token not in token_crops:
            token_crops[token] = []
        if len(token_crops[token]) < 4:
            token_crops[token].append(crop.copy())

# Save a contact sheet per token
for token, crops in token_crops.items():
    sheet = np.hstack(crops)
    cv2.imwrite(f'token_crops/{token}.jpg', sheet)
    print(f'Token {token}: {len(crops)} sample crops saved')

print(f'\nTotal unique tokens: {len(gallery)}')
print('Open token_crops/ folder to visually verify each person')
