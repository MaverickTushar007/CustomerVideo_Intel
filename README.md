# 🎯 CustomerVideo Intel — Restaurant Analytics Pipeline

A real-time computer vision pipeline for restaurant customer intelligence. Uses YOLO person detection, multi-object tracking, multimodal staff identification, and an LLM-powered natural language query agent to extract actionable analytics from CCTV footage.

---

## 🚀 Features

- **Person Detection** — YOLOv11 for accurate real-time person detection
- **Multi-Object Tracking** — Custom `PositionTracker` with centroid-based IoU tracking
- **Multimodal Staff Identification** — 3-model fusion:
  - 🎨 Uniform color detection (HSV torso scan — red, blue, black ranges)
  - 🏷️ Badge/name-tag detection via contour analysis
  - 🧠 Re-ID embedding matcher (with enrolled staff embeddings)
- **Proximity-Based Service Tracking** — Detects when staff attends a customer (centroid distance < 120px)
- **Color-Coded Visualization** — Green=Staff, Cyan=Served Customer, Orange=Waiting Customer
- **SQLite Analytics DB** — Persists visit events, dwell times, staff flags, and service latency
- **LLM Query Agent** — Natural language → SQL via Llama-3.3-70b on Groq
- **FastAPI REST Layer** — Exposes analytics via HTTP API

---

## 📁 Project Structure

```
CustomerVideo_Intel/
├── pipeline_position.py         # Main tracking + analytics pipeline
├── run_pipeline.sh              # One-command runner script
│
├── restaurant_analytics/        # Core analytics package
│   ├── staff_identifier.py      # Multimodal staff classifier (Uniform + Badge + Re-ID)
│   ├── schema.py                # Event JSON schema models
│   ├── interfaces.py            # Abstract interfaces
│   ├── zone_mapper.py           # Floorplan zone mapping
│   └── edge_agent.py            # Edge Agent process loop
│
├── agent/
│   └── query_agent.py           # Groq LLM NL→SQL query engine
│
├── api/
│   └── main.py                  # FastAPI REST endpoints
│
├── db/
│   └── setup.py                 # SQLite schema setup
│
├── ingestion/
│   └── frame_sampler.py         # Video frame stream sampler
│
├── tracking/
│   └── position_tracker.py      # Centroid-based multi-object tracker
│
├── tests/
│   └── test_restaurant_analytics.py  # Unit tests for analytics modules
│
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up the database

```bash
python db/setup.py
```

### 3. Set your Groq API key

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com)

### 4. Run the pipeline

```bash
./run_pipeline.sh your_video.mp4
```

This will:
1. Clear previous database entries
2. Run YOLO + tracker + staff classifier on the video
3. Auto-run 13 NL analytics queries via the LLM agent

---

## 🧠 Staff Identification — How It Works

```
Frame → YOLO BBox → Torso Crop → HSV Mask ──► Uniform Confidence
                 └─► Chest Crop → Contour Analysis ──► Badge Confidence
                                                              │
                                              MultiModalStaffIdentifier
                                              (weighted fusion ≥ 0.30 threshold)
                                                              │
                                              STAFF ✅ or CUSTOMER ❌
```

### Tunable Parameters

| Parameter | Default | Description |
|---|---|---|
| `pixel_ratio_threshold` | 0.25 | % of torso pixels matching uniform color |
| `STAFF_CONFIDENCE_THRESHOLD` | 0.30 | Min confidence to classify as staff |
| `PROXIMITY_THRESHOLD` | 120px | Staff-customer distance for "attend" event |

---

## 📊 Sample Analytics Output

```
Q: How many total visitors have we had?        → 17
Q: How many were customers (not staff)?        → 13
Q: How many staff were active?                 → 4
Q: What % of customers abandoned?             → 30.8%
Q: How many were successfully served?          → 6
Q: Avg time before customer attended by staff? → 3.20s
Q: Max wait time before service?               → 6.11s
```

---

## 🏗️ Architecture

```
Video File
    │
    ▼
Frame Sampler (fps_target=8)
    │
    ▼
YOLO Detector (yolo11n.pt, conf=0.25)
    │
    ▼
Position Tracker (centroid IoU)
    │
    ├──► MultiModalStaffIdentifier
    │         ├── UniformColorIdentifier (HSV)
    │         └── BadgeDetector (contours)
    │
    ├──► Proximity Attend Tracker (120px threshold)
    │
    ▼
SQLite DB (persons + wait_metrics)
    │
    ▼
Groq LLM Query Agent (Llama-3.3-70b)
    │
    ▼
Analytics Report
```

---

## 🧪 Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

---

## 📦 Requirements

- Python 3.10+
- OpenCV
- Ultralytics YOLO
- Groq Python SDK
- FastAPI + Uvicorn
- SQLite3 (built-in)

See `requirements.txt` for pinned versions.

---

## ⚠️ Notes

- YOLO model weights (`.pt` files) are **not included** — download from [Ultralytics](https://docs.ultralytics.com/models/)
- Video files are **not included** in the repo (add your own CCTV footage)
- Set `GROQ_API_KEY` before running — never hardcode it

---

## 📄 License

MIT
