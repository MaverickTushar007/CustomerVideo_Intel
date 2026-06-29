from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.query_agent import ask
import sqlite3

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class NLQuery(BaseModel):
    question: str

@app.post("/ask")
def query(q: NLQuery):
    result = ask(q.question)
    return {"result": result}

@app.get("/cameras")
def cameras():
    db = sqlite3.connect("db/customer_intel.db")
    rows = db.execute("SELECT DISTINCT camera_id FROM persons ORDER BY camera_id").fetchall()
    db.close()
    return {"cameras": ["all"] + [r[0] for r in rows]}

@app.get("/stats")
def stats(camera: str = Query(default="all")):
    db = sqlite3.connect("db/customer_intel.db")
    cam_filter = "" if camera == "all" else f"AND p.camera_id = '{camera}'"
    cam_filter_w = "" if camera == "all" else f"AND w.token_id IN (SELECT token_id FROM persons WHERE camera_id = '{camera}')"

    persons_raw = db.execute(f"""
        SELECT p.token_id, p.first_seen, p.last_seen,
               p.camera_id, w.wait_seconds, w.abandoned
        FROM persons p
        LEFT JOIN wait_metrics w ON p.token_id = w.token_id
        WHERE 1=1 {cam_filter}
        ORDER BY w.wait_seconds DESC
    """).fetchall()

    persons = [
        {{"token_id": r[0], "first_seen": r[1], "last_seen": r[2],
          "camera_id": r[3], "wait_seconds": r[4], "abandoned": r[5]}}
        for r in persons_raw
    ]

    data = {{
        "total_visitors": db.execute(f"SELECT COUNT(*) FROM persons WHERE 1=1 {cam_filter}").fetchone()[0],
        "avg_dwell": db.execute(f"SELECT ROUND(AVG(wait_seconds),1) FROM wait_metrics w WHERE wait_seconds > 2 {cam_filter_w}").fetchone()[0],
        "max_dwell": db.execute(f"SELECT ROUND(MAX(wait_seconds),1) FROM wait_metrics w WHERE 1=1 {cam_filter_w}").fetchone()[0],
        "abandonment_rate": db.execute(f"SELECT ROUND(100.0 * SUM(abandoned) / COUNT(*), 1) FROM wait_metrics w WHERE wait_seconds > 2 {cam_filter_w}").fetchone()[0],
        "persons": persons
    }}
    db.close()
    return data

import threading
from api.process import start_job

DEMO_VIDEOS = {
    "retail_store": "vidssave.com HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store 720p.mp4",
    "cafe": "vidssave.com Surya Security Cctv Hik 2MP _ 1080p  cafe TP Surabaya 20170623 1080P.mp4",
    "midtown": "vidssave.com Midtown corner store surveillance video 11-25-18 720P.mp4",
    "retail_usa": "vidssave.com E43A inside the Retail store in USA #2 1080P.mp4",
}

@app.post("/demo/{video_key}")
def run_demo(video_key: str):
    if video_key not in DEMO_VIDEOS:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Demo not found")
    path = DEMO_VIDEOS[video_key]
    import os
    if not os.path.exists(path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Video file not found: {path}")
    job_id = start_job(path)
    return {"job_id": job_id}

import os
import uuid
from fastapi import UploadFile, File, HTTPException

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(saved_path, "wb") as f:
        f.write(await file.read())
    job_id = start_job(saved_path)
    return {"job_id": job_id}


class UrlUpload(BaseModel):
    url: str

@app.post("/upload/url")
def upload_url(body: UrlUpload):
    import yt_dlp
    base = os.path.join(UPLOAD_DIR, uuid.uuid4().hex)
    ydl_opts = {"outtmpl": base + ".%(ext)s", "format": "mp4/best", "quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(body.url, download=True)
            saved_path = ydl.prepare_filename(info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")
    job_id = start_job(saved_path)
    return {"job_id": job_id}
