# ingestion/frame_sampler.py

import cv2
import numpy as np

def apply_clahe(frame):
    """Normalize lighting — handles dim restaurants and bright casinos."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def stream_frames(source, fps_target=8):
    """
    source: int (webcam), RTSP URL string, or video file path
    Yields (frame_id, timestamp, frame) at target FPS.
    """
    cap = cv2.VideoCapture(source)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, int(native_fps / fps_target))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % skip == 0:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield frame_id, ts, apply_clahe(frame)
        frame_id += 1

    cap.release()