import numpy as np
import uuid

class PositionTracker:
    def __init__(self, max_distance=150, max_missing_frames=150):
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1+x2)/2, (y1+y2)/2)

    def update(self, detections, frame_id, ts):
        results = []
        matched_tokens = set()

        for bbox in detections:
            centroid = self._centroid(bbox)
            best_token, best_dist = None, float('inf')

            for token, track in self.tracks.items():
                if token in matched_tokens:
                    continue
                dist = np.sqrt(
                    (centroid[0]-track['centroid'][0])**2 +
                    (centroid[1]-track['centroid'][1])**2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_token = token

            if best_token and best_dist < self.max_distance:
                self.tracks[best_token].update({
                    'centroid': centroid, 'bbox': bbox,
                    'last_frame': frame_id, 'last_ts': ts
                })
                matched_tokens.add(best_token)
                results.append((best_token, bbox, False))
            else:
                token = str(uuid.uuid4())[:8]
                self.tracks[token] = {
                    'centroid': centroid, 'bbox': bbox,
                    'last_frame': frame_id, 'entry_ts': ts,
                    'last_ts': ts
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
        return [(t, tr['entry_ts'], tr['last_ts']) for t, tr in self.tracks.items()]
