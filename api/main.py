from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import uuid
from groq import Groq
import os

# Load .env file if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set externally

# Ensure DB exists on startup (Railway ephemeral filesystem)
import os
os.makedirs("db", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

import sqlite3 as _sqlite3
_conn = _sqlite3.connect("db/customer_intel.db")
_conn.executescript("""
CREATE TABLE IF NOT EXISTS persons (
    token_id TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    last_seen TEXT,
    camera_id TEXT,
    abandoned INTEGER DEFAULT 0,
    is_staff INTEGER DEFAULT 0,
    staff_id TEXT
);
CREATE TABLE IF NOT EXISTS wait_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id TEXT,
    entry_time TEXT,
    exit_time TEXT,
    wait_seconds REAL,
    time_to_service REAL,
    abandoned INTEGER DEFAULT 0,
    date TEXT
);
""")
_conn.commit()
_conn.close()
app = FastAPI(title="Customer Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_db():
    return sqlite3.connect('db/customer_intel.db')

SCHEMA = """
Tables:
- persons(token_id TEXT, first_seen TEXT, last_seen TEXT, camera_id TEXT, abandoned INTEGER, is_staff INTEGER, staff_id TEXT)
- wait_metrics(id INTEGER, token_id TEXT, entry_time TEXT, exit_time TEXT, wait_seconds REAL, time_to_service REAL, abandoned INTEGER, date TEXT)

Notes:
- abandoned=1 means left without being served
- is_staff=1 means person is a staff member, is_staff=0 means customer
- staff_id is the unique identifier of staff members (e.g. emp_mary, emp_jack)
- wait_seconds is dwell time in SECONDS (not minutes)
- time_to_service is the time in seconds from when the customer entered to when they were first attended/addressed by a staff member (null if not attended or if staff member)
- The date column only exists in wait_metrics
- entry_time and exit_time are ISO8601 UTC strings like 2026-06-27T08:30:10+00:00
- To filter by today: WHERE date = date('now')
- To filter by hour: WHERE strftime('%H', entry_time) = '20' (for 8pm UTC)
- To filter a time range: WHERE entry_time BETWEEN '2026-06-27T08:00:00' AND '2026-06-27T09:00:00'
- Always filter with wait_seconds > 3 to exclude false detections
- To count visitors in last N minutes: WHERE entry_time >= datetime('now', '-N minutes')
- Convert seconds to minutes by dividing by 60 when answering
"""

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/metrics/summary")
def summary():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT COUNT(*), ROUND(AVG(wait_seconds),1), ROUND(MAX(wait_seconds),1),
               SUM(CASE WHEN abandoned=1 THEN 1 ELSE 0 END)
        FROM wait_metrics
        WHERE wait_seconds > 3
    """)
    r = cur.fetchone()
    db.close()
    total = r[0] or 0
    abandoned = r[3] or 0
    return {
        "total_visitors": total,
        "avg_dwell_seconds": r[1],
        "max_dwell_seconds": r[2],
        "abandoned_count": abandoned,
        "abandonment_rate_pct": round((abandoned / total) * 100, 1) if total else 0
    }

@app.get("/metrics/persons")
def all_persons():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT p.token_id, p.first_seen, p.camera_id, w.wait_seconds, w.abandoned
        FROM persons p
        LEFT JOIN wait_metrics w ON p.token_id = w.token_id
        WHERE w.wait_seconds > 3
        ORDER BY p.first_seen
    """)
    rows = cur.fetchall()
    db.close()
    return [{"token_id": r[0], "entered": r[1], "camera": r[2],
             "dwell_seconds": r[3], "abandoned": bool(r[4])} for r in rows]

@app.get("/metrics/longest_waits")
def longest_waits(limit: int = 5):
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT token_id, wait_seconds, entry_time, abandoned
        FROM wait_metrics
        WHERE wait_seconds > 3
        ORDER BY wait_seconds DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    db.close()
    return [{"token_id": r[0], "dwell_seconds": r[1],
             "entered": r[2], "abandoned": bool(r[3])} for r in rows]

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(body: Question):
    db = get_db()
    sql_msg = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You write SQLite SELECT queries for this schema:\n{SCHEMA}\nReturn ONLY raw SQL, no markdown, no explanation. Always filter with wait_seconds > 3."},
            {"role": "user", "content": body.question}
        ],
        max_tokens=200, temperature=0
    )
    sql = sql_msg.choices[0].message.content.strip()
    try:
        cur = db.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        result = [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        db.close()
        return {"question": body.question, "sql": sql,
                "plain_answer": f"Sorry, could not answer that. ({e})", "result": []}
    finally:
        db.close()

    plain_msg = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a friendly business analyst explaining venue analytics to a restaurant owner. Give a clear 1-2 sentence answer. Rules: (1) wait_seconds is in SECONDS, convert to minutes by dividing by 60. (2) Mention the time period clearly. (3) Never mention token IDs or SQL. (4) If no data found, say so clearly."},
            {"role": "user", "content": f"Question: {body.question}\nData: {result}"}
        ],
        max_tokens=150, temperature=0.3
    )
    return {"question": body.question, "sql": sql, "result": result,
            "plain_answer": plain_msg.choices[0].message.content.strip()}

from fastapi import UploadFile, File, BackgroundTasks
import shutil, os
from api.process import start_job, jobs

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    path = f"{UPLOAD_DIR}/{uuid.uuid4()}.{ext}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = start_job(path)
    return {"job_id": job_id}

@app.post("/upload/url")
async def upload_url(body: dict):
    import subprocess, uuid as _uuid
    path = f"{UPLOAD_DIR}/{_uuid.uuid4()}.mp4"
    result = subprocess.run(
        ["yt-dlp", body["url"], "-o", path, "--no-playlist"],
        capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": "Failed to download video"}
    job_id = start_job(path)
    return {"job_id": job_id}

@app.get("/job/{job_id}")
def job_status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})


@app.get("/metrics/hourly")
def hourly():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT
            CAST(strftime('%H', entry_time) AS INTEGER) AS hour,
            COUNT(*) AS visitors,
            ROUND(AVG(wait_seconds), 1) AS avg_dwell,
            SUM(CASE WHEN abandoned=1 THEN 1 ELSE 0 END) AS abandoned
        FROM wait_metrics
        WHERE wait_seconds > 3
        GROUP BY hour
        ORDER BY hour
    """)
    rows = cur.fetchall()
    db.close()
    return [{"hour": r[0], "visitors": r[1],
             "avg_dwell_seconds": r[2], "abandoned": r[3]} for r in rows]

@app.get("/metrics/business_iq")
def business_iq():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN abandoned=0 THEN 1 ELSE 0 END) as served,
            AVG(wait_seconds) as avg_dwell,
            MAX(wait_seconds) as max_dwell
        FROM wait_metrics
        WHERE wait_seconds > 3
    """)
    r = cur.fetchone()
    db.close()

    total = r[0] or 0
    served = r[1] or 0
    avg_dwell = r[2] or 0
    
    if total == 0:
        return {"score": 0, "grade": "N/A", "breakdown": {}}

    # Service rate score (0-100, 40% weight)
    service_rate = (served / total) * 100
    service_score = service_rate

    # Dwell score (0-100, 30% weight)
    # Sweet spot: 5-15 min (300-900s). Too short = bad service, too long = inefficiency
    if avg_dwell < 60:
        dwell_score = (avg_dwell / 60) * 40
    elif avg_dwell <= 900:
        dwell_score = 40 + ((avg_dwell - 60) / 840) * 60
    else:
        dwell_score = max(0, 100 - ((avg_dwell - 900) / 300) * 20)

    # Abandonment score (0-100, 30% weight)
    abandonment_rate = ((total - served) / total) * 100
    abandonment_score = max(0, 100 - abandonment_rate)

    # Weighted final score
    final = (service_score * 0.4) + (dwell_score * 0.3) + (abandonment_score * 0.3)
    final = round(min(100, max(0, final)), 1)

    if final >= 80:
        grade, color = "A", "#10b981"
    elif final >= 65:
        grade, color = "B", "#3b82f6"
    elif final >= 50:
        grade, color = "C", "#f59e0b"
    elif final >= 35:
        grade, color = "D", "#f97316"
    else:
        grade, color = "F", "#ef4444"

    return {
        "score": final,
        "grade": grade,
        "color": color,
        "breakdown": {
            "service_rate_pct": round(service_rate, 1),
            "avg_dwell_seconds": round(avg_dwell, 1),
            "abandonment_rate_pct": round(abandonment_rate, 1),
            "service_score": round(service_score, 1),
            "dwell_score": round(dwell_score, 1),
            "abandonment_score": round(abandonment_score, 1)
        },
        "insights": [
            "Staff attendance is the biggest opportunity for improvement" if service_rate < 50
            else "Good staff coverage — maintain current floor staffing",
            f"Average visit lasts {round(avg_dwell)}s — {'consider faster table service' if avg_dwell > 900 else 'within healthy range'}",
            f"{round(abandonment_rate)}% of visitors left without being served"
        ]
    }
