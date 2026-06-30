from ingestion.frame_sampler import stream_frames
from ultralytics import YOLO
import torchreid
import torch
import numpy as np
import cv2

# Load models
detector = YOLO('yolo11n.pt')
embedder = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
embedder.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder.to(device)

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

frame_count = 0
for fid, ts, frame in stream_frames('test_video.mp4', fps_target=2):
    results = detector(frame, conf=0.3, classes=[0], verbose=False)
    boxes = results[0].boxes
    print(f'Frame {fid} | {len(boxes)} people')
    for i, box in enumerate(boxes):
        emb = get_embedding(frame, box.xyxy[0].tolist())
        if emb is not None:
            print(f'  Person {i}: embedding shape {emb.shape}, norm {np.linalg.norm(emb):.4f}')
    frame_count += 1
    if frame_count >= 3:
        break
print('OSNet Re-ID working!')
