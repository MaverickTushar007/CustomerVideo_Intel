import numpy as np
import uuid
import torch
import torchvision.transforms as T
from torchvision.models import resnet18

class DeepTracker:
    def __init__(self, max_distance=80, max_missing_frames=100, reid_threshold=0.60):
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.reid_threshold = reid_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_reid()

    def _load_reid(self):
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def _embedding(self, frame, bbox):
        x1,y1,x2,y2 = [int(v) for v in bbox]
        crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            return None
        t = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(t).cpu().numpy()[0]

    def _cosine(self, a, b):
        if a is None or b is None:
            return 1.0
        return 1 - np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

    def _centroid(self, bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)/2, (y1+y2)/2)

    def _iou(self, b1, b2):
        ix1,iy1 = max(b1[0],b2[0]), max(b1[1],b2[1])
        ix2,iy2 = min(b1[2],b2[2]), min(b1[3],b2[3])
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-8)

    def update(self, detections, frame_id, ts, frame=None):
        results = []
        matched_tokens = set()

        for bbox in detections:
            centroid = self._centroid(bbox)
            emb = self._embedding(frame, bbox) if frame is not None else None

            best_token, best_score = None, float('inf')
            for token, track in self.tracks.items():
                if token in matched_tokens:
                    continue
                dist = np.sqrt(
                    (centroid[0]-track['centroid'][0])**2 +
                    (centroid[1]-track['centroid'][1])**2
                )
                iou = self._iou(bbox, track['bbox'])
                cos = self._cosine(emb, track.get('embedding'))
                score = 0.4*min(dist/self.max_distance,1.0) + 0.3*(1-iou) + 0.3*cos
                if score < best_score:
                    best_score = score
                    best_token = token

            if best_token and best_score < self.reid_threshold:
                self.tracks[best_token].update({
                    'centroid': centroid, 'bbox': bbox,
                    'last_frame': frame_id, 'last_ts': ts,
                    'embedding': emb
                })
                matched_tokens.add(best_token)
                results.append((best_token, bbox, False))
            else:
                token = str(uuid.uuid4())[:8]
                self.tracks[token] = {
                    'centroid': centroid, 'bbox': bbox,
                    'last_frame': frame_id, 'entry_ts': ts,
                    'last_ts': ts, 'embedding': emb
                }
                matched_tokens.add(token)
                results.append((token, bbox, True))

        return results

    def get_exited(self, frame_id):
        exited = []
        for token, track in list(self.tracks.items()):
            if (frame_id - track['last_frame']) > self.max_missing_frames:
                exited.append((token, track['entry_ts'], track['last_ts']))
                del self.tracks[token]
        return exited

    def flush_all(self):
        return [(t, tr['entry_ts'], tr['last_ts']) for t,tr in self.tracks.items()]
