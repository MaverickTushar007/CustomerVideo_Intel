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
CREATE TABLE IF NOT EXISTS venue_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id TEXT DEFAULT 'default',
    metric_type TEXT,
    day_of_week INTEGER,
    hour_of_day INTEGER,
    value REAL,
    sample_count INTEGER DEFAULT 1,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS query_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id TEXT DEFAULT 'default',
    question TEXT,
    sql_used TEXT,
    plain_answer TEXT,
    rating INTEGER DEFAULT 0,
    created_at TEXT
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
def ask(body: Question, venue_id: str = "default"):
    from db.memory_updater import build_context_snapshot
    import uuid as _uuid_mod
    from datetime import datetime, timezone as _tz

    # Build context-aware system prompt with venue memory + few-shot examples
    venue_context = build_context_snapshot(venue_id=venue_id)
    system_sql = (
        f"You write SQLite SELECT queries for this schema:\n{SCHEMA}\n"
        f"Return ONLY raw SQL, no markdown, no explanation. "
        f"Always filter with wait_seconds > 3.\n\n"
        f"{venue_context}"
    )

    db = get_db()
    sql_msg = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_sql},
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
                "plain_answer": f"Sorry, could not answer that. ({e})",
                "result": [], "answer_id": None, "anomaly_flags": []}

    system_plain = (
        "You are a friendly business analyst explaining venue analytics to a restaurant owner. "
        "Give a clear 1-2 sentence answer. Rules: "
        "(1) wait_seconds is in SECONDS, convert to minutes by dividing by 60. "
        "(2) Mention the time period clearly. "
        "(3) Never mention token IDs or SQL. "
        "(4) If venue memory baselines show anomalies, mention them naturally. "
        "(5) If no data found, say so clearly.\n\n"
        f"{venue_context}"
    )
    plain_msg = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_plain},
            {"role": "user", "content": f"Question: {body.question}\nData: {result}"}
        ],
        max_tokens=200, temperature=0.3
    )
    plain_answer = plain_msg.choices[0].message.content.strip()

    # Persist Q&A for feedback / few-shot learning
    answer_id = str(_uuid_mod.uuid4())[:12]
    try:
        cur2 = db.cursor()
        cur2.execute("""
            INSERT INTO query_examples
              (id, venue_id, question, sql_used, plain_answer, rating, created_at)
            VALUES (?,?,?,?,?,0,?)
        """, (answer_id, venue_id, body.question, sql, plain_answer,
              datetime.now(_tz.utc).isoformat()))
        db.commit()
    except Exception:
        pass
    finally:
        db.close()

    # Extract anomaly flags for the dashboard
    anomaly_flags = [
        line.strip().lstrip("• ")
        for line in venue_context.split("\n")
        if line.strip().startswith("•") and "ANOMAL" not in venue_context.split(line)[0].upper()[-20:]
        and "above baseline" in line or "below baseline" in line
    ]

    return {
        "question": body.question,
        "sql": sql,
        "result": result,
        "plain_answer": plain_answer,
        "answer_id": answer_id,
        "anomaly_flags": anomaly_flags,
        "venue_id": venue_id,
    }

from fastapi import UploadFile, File, BackgroundTasks
import shutil, os
from api.process import start_job, jobs


class RatingBody(BaseModel):
    rating: int  # 1 = thumbs up, -1 = thumbs down


@app.post("/ask/{answer_id}/rate")
def rate_answer(answer_id: str, body: RatingBody):
    """Store owner feedback on an answer — grows the venue's few-shot example pool."""
    if body.rating not in (1, -1):
        return {"error": "rating must be 1 (good) or -1 (bad)"}
    db = get_db()
    cur = db.cursor()
    cur.execute("UPDATE query_examples SET rating=? WHERE id=?",
                (body.rating, answer_id))
    db.commit()
    affected = cur.rowcount
    db.close()
    if affected == 0:
        return {"error": "answer_id not found"}
    return {"status": "ok", "answer_id": answer_id, "rating": body.rating}

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


@app.get("/metrics/baseline")
def baseline(venue_id: str = "default"):
    """Returns learned baselines vs today's live data with anomaly flags."""
    from db.memory_updater import build_context_snapshot, _detect_anomalies, _connect
    from datetime import datetime, timezone as _tz
    import calendar

    db_conn = _connect()
    cur = db_conn.cursor()
    now = datetime.now(_tz.utc)
    sqlite_dow = (now.weekday() + 1) % 7
    current_hour = now.hour

    # Pull all baselines for this venue
    cur.execute("""
        SELECT metric_type, day_of_week, hour_of_day, value, sample_count
        FROM venue_memory
        WHERE venue_id=?
        ORDER BY day_of_week, hour_of_day
    """, (venue_id,))
    memory_rows = cur.fetchall()

    # Live stats for today
    cur.execute("""
        SELECT
            COUNT(*) as visitors,
            AVG(wait_seconds) as avg_dwell,
            ROUND(100.0*SUM(CASE WHEN abandoned=1 THEN 1 ELSE 0 END)/COUNT(*),1) as abandon_rate
        FROM wait_metrics
        WHERE wait_seconds > 3 AND date = date('now')
    """)
    live = cur.fetchone()

    anomalies = _detect_anomalies(cur, venue_id, sqlite_dow, current_hour)
    db_conn.close()

    baselines = [
        {"metric_type": r[0], "day_of_week": r[1], "hour_of_day": r[2],
         "value": r[3], "sample_count": r[4]}
        for r in memory_rows
    ]

    dow_name = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][sqlite_dow]
    return {
        "venue_id": venue_id,
        "current_slot": {"day_of_week": dow_name, "hour": current_hour},
        "live_today": {
            "visitors": live[0] if live else 0,
            "avg_dwell_seconds": round(live[1], 1) if live and live[1] else None,
            "abandonment_rate_pct": live[2] if live else None,
        },
        "baselines": baselines,
        "anomaly_flags": anomalies,
        "has_memory": len(baselines) > 0,
        "runs_recorded": max((r["sample_count"] for r in baselines), default=0),
    }


@app.get("/venues")
def list_venues():
    """Lists all venues that have learned memory."""
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT venue_id,
               COUNT(*) as memory_slots,
               MAX(sample_count) as max_runs,
               MAX(updated_at) as last_updated
        FROM venue_memory
        GROUP BY venue_id
    """)
    rows = cur.fetchall()
    db.close()
    return [
        {"venue_id": r[0], "memory_slots": r[1],
         "runs_recorded": r[2], "last_updated": r[3]}
        for r in rows
    ]


@app.get("/metrics/compare")
def compare_venues(venue_a: str, venue_b: str):
    """Cross-venue comparison of learned metric averages."""
    db = get_db()
    cur = db.cursor()
    result = {}
    for vid in (venue_a, venue_b):
        cur.execute("""
            SELECT metric_type, AVG(value), SUM(sample_count)
            FROM venue_memory WHERE venue_id=?
            GROUP BY metric_type
        """, (vid,))
        rows = cur.fetchall()
        result[vid] = {
            r[0]: {"avg": round(r[1], 2), "total_samples": r[2]}
            for r in rows
        }
    db.close()
    return {"comparison": result, "venues": [venue_a, venue_b]}


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
