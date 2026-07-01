import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from ingestion.frame_sampler import stream_frames

# Sequential Tracker for clean sequential IDs (#1, #2, #3...)
class SequentialPositionTracker:
    def __init__(self, max_distance=150, max_missing_frames=150):
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.next_id = 1

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
                token = self.next_id
                self.next_id += 1
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


# Colors for drawing
COLOR_GUEST = (0, 165, 255)  # Orange/amber for guests
COLOR_STAFF = (0, 255, 0)    # Green for staff
COLOR_TEXT = (255, 255, 255)
COLOR_BG_STATS = (30, 30, 30)

def main():
    video_path = "Dark_lighting.mp4"
    output_path = "output_dark_lighting.mp4"

    print("Loading YOLO model...")
    # Load custom staff/customer classifier YOLO model
    detector = YOLO('yolo_staff_customer.pt')
    
    # Select best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Initialize Sequential Tracker
    tracker = SequentialPositionTracker(max_distance=180, max_missing_frames=120)
    
    # We will keep track of class votes for each tracking ID
    # 0 = customer/guest, 1 = staff
    token_class_votes = {}
    token_final_class = {}

    # Cumulative sets to keep track of total unique guests and staff
    cumulative_guests = set()
    cumulative_staff = set()

    writer = None
    frame_count = 0
    fps_target = 8

    print(f"Processing {video_path}...")
    for fid, ts, frame in stream_frames(video_path, fps_target=fps_target):
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps_target, (w, h))

        # Detect
        results = detector(frame, conf=0.25, verbose=False, device=device)
        bboxes = []
        box_classes = []
        box_confidences = []
        for box in results[0].boxes:
            bboxes.append(box.xyxy[0].tolist())
            box_classes.append(int(box.cls[0]))
            box_confidences.append(float(box.conf[0]))

        # Update tracker
        tracks = tracker.update(bboxes, fid, ts)

        # Clean up exited tracks from the active tracker to prevent ID hijacking by new detections
        tracker.get_exited(fid)

        # Class vote update
        for i, (token, bbox, is_new) in enumerate(tracks):
            cls = box_classes[i]
            token_class_votes.setdefault(token, []).append(cls)
            votes = token_class_votes[token]
            
            # Majority vote
            is_staff = (votes.count(1) > votes.count(0))
            token_final_class[token] = "staff" if is_staff else "guest"

            if is_staff:
                cumulative_staff.add(token)
                # If they were previously in cumulative_guests, remove them to avoid double counting
                cumulative_guests.discard(token)
            else:
                if token not in cumulative_staff:
                    cumulative_guests.add(token)

        # Draw overlays
        for token, bbox, is_new in tracks:
            role = token_final_class.get(token, "guest")
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Label format: #1guest or #6staff
            label_text = f"#{token}{role}"
            color = COLOR_STAFF if role == "staff" else COLOR_GUEST

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(y1 - 20, 0)), (x1 + tw + 6, max(y1 - 2, 0)), color, -1)
            # Draw label text
            cv2.putText(frame, label_text, (x1 + 3, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw beautiful dashboard overlay on top-left of the screen
        # Height: 100, Width: 410
        cv2.rectangle(frame, (10, 10), (420, 110), COLOR_BG_STATS, -1)
        cv2.rectangle(frame, (10, 10), (420, 110), (100, 100, 100), 2)
        
        # Calculate counts
        # Covers = Total unique guests seen so far
        # Staff = Total unique staff seen so far
        n_covers = len(cumulative_guests)
        n_staff = len(cumulative_staff)
        
        # Current active in frame
        active_guests = sum(1 for token, _, _ in tracks if token_final_class.get(token) == "guest")
        active_staff = sum(1 for token, _, _ in tracks if token_final_class.get(token) == "staff")

        # Texts
        title_text = "CUSTOMER INTELLIGENCE LIVE SCAN"
        time_text = f"Video Time: {ts:.1f}s | Frame: {fid}"
        counts_text1 = f"Covers (Guests): {n_covers}  |  Staff: {n_staff}"
        counts_text2 = f"Active in Frame: {active_guests} Guest(s), {active_staff} Staff"

        cv2.putText(frame, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, counts_text1, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, counts_text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, time_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        # Write frame to video
        writer.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames... {ts:.1f}s. Covers: {n_covers}, Staff: {n_staff}")

    if writer is not None:
        writer.release()
    print(f"Successfully processed {frame_count} frames. Output saved to {output_path}")
    print(f"Final cumulative counts -> Covers (Guests): {len(cumulative_guests)}, Staff: {len(cumulative_staff)}")

if __name__ == "__main__":
    main()
