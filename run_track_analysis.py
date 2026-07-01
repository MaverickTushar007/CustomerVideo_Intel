import cv2
import numpy as np
import torch
from ultralytics import YOLO
from run_dark_test import SequentialPositionTracker
from ingestion.frame_sampler import stream_frames

def main():
    tracker = SequentialPositionTracker(max_distance=180, max_missing_frames=120)
    detector = YOLO('yolo_staff_customer.pt')

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    tracks_history = {} # id -> list of (frame_id, ts, centroid, role)
    token_class_votes = {}

    for fid, ts, frame in stream_frames('Dark_lighting.mp4', fps_target=8):
        results = detector(frame, conf=0.25, verbose=False, device=device)
        bboxes = []
        box_classes = []
        for box in results[0].boxes:
            bboxes.append(box.xyxy[0].tolist())
            box_classes.append(int(box.cls[0]))
        
        tracks = tracker.update(bboxes, fid, ts)
        tracker.get_exited(fid)  # Clean up exited tracks!
        
        for i, (token, bbox, is_new) in enumerate(tracks):
            cls = box_classes[i]
            token_class_votes.setdefault(token, []).append(cls)
            role = 'staff' if (token_class_votes[token].count(1) > token_class_votes[token].count(0)) else 'guest'
            
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            tracks_history.setdefault(token, []).append((fid, ts, (cx, cy), role))

    print('\nTrack analysis:')
    for token, history in sorted(tracks_history.items()):
        start_fid, start_ts = history[0][0], history[0][1]
        end_fid, end_ts = history[-1][0], history[-1][1]
        duration = end_ts - start_ts
        roles = [h[3] for h in history]
        final_role = 'staff' if roles.count('staff') > roles.count('guest') else 'guest'
        print(f'ID {token:2d}: type={final_role:5s} duration={duration:5.1f}s frames={len(history):4d} ({start_ts:.1f}s to {end_ts:.1f}s)')

if __name__ == "__main__":
    main()
