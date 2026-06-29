import cv2
import numpy as np
import os

def analyze_video_sample(video_path, sample_frames=10):
    """
    Samples the first N frames to assess video characteristics
    and returns optimal processing parameters.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    brightness_values = []
    contrast_values = []

    for i in range(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * (total_frames // sample_frames))
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray))
        contrast_values.append(np.std(gray))

    cap.release()

    avg_brightness = np.mean(brightness_values) if brightness_values else 128
    avg_contrast = np.mean(contrast_values) if contrast_values else 50

    # ── Parameter decisions ───────────────────────────────────────────────

    # Lighting: dim venue needs lower conf threshold + stronger CLAHE
    if avg_brightness < 60:
        conf_threshold = 0.20
        clahe_clip = 3.0
        lighting = "dim"
    elif avg_brightness < 100:
        conf_threshold = 0.25
        clahe_clip = 2.0
        lighting = "moderate"
    else:
        conf_threshold = 0.30
        clahe_clip = 1.5
        lighting = "bright"

    # Duration: longer video = lower FPS to save compute
    if duration < 60:
        fps_target = 12
    elif duration < 300:
        fps_target = 8
    else:
        fps_target = 5

    # Re-ID threshold: overhead CCTV of seated people needs more lenient threshold
    # OSNet trained on walking pedestrians, not overhead seated views
    if avg_contrast < 30:
        reid_threshold = 0.50
    elif avg_contrast < 50:
        reid_threshold = 0.55
    else:
        reid_threshold = 0.58

    params = {
        "conf_threshold": conf_threshold,
        "reid_threshold": reid_threshold,
        "fps_target": fps_target,
        "clahe_clip": clahe_clip,
        "lighting": lighting,
        "avg_brightness": round(float(avg_brightness), 1),
        "avg_contrast": round(float(avg_contrast), 1),
        "video_duration_seconds": round(duration, 1),
        "reasoning": {
            "lighting": f"Brightness {round(avg_brightness)} → {lighting} environment → conf={conf_threshold}",
            "reid": f"Contrast {round(avg_contrast)} → reid_threshold={reid_threshold}",
            "fps": f"Duration {round(duration)}s → fps_target={fps_target}"
        }
    }

    return params


if __name__ == "__main__":
    import json, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_video2.mp4"
    if os.path.exists(path):
        result = analyze_video_sample(path)
        print(json.dumps(result, indent=2))
    else:
        print(f"Video not found: {path}")
