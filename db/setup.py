import sqlite3, os

db_dir = 'db'
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, 'customer_intel.db')

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Enable foreign keys
cur.execute("PRAGMA foreign_keys = ON;")

cur.executescript("""
DROP TABLE IF EXISTS raw_observations;
DROP TABLE IF EXISTS temporal_sessions;
DROP TABLE IF EXISTS staff_resolutions;
DROP TABLE IF EXISTS business_events;
DROP TABLE IF EXISTS venue_memory;
DROP TABLE IF EXISTS query_examples;

-- 1. RAW OBSERVATIONS
CREATE TABLE raw_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    confidence REAL,
    embedding BLOB
);

-- 2. TEMPORAL SESSIONS
CREATE TABLE temporal_sessions (
    session_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    duration_seconds REAL
);

-- 3. STAFF RESOLUTIONS (Decoupled semantic staff mapping)
CREATE TABLE staff_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES temporal_sessions(session_id) ON DELETE CASCADE,
    staff_id TEXT,
    confidence REAL,
    resolution_method TEXT,
    resolved_at TEXT
);

-- 4. DERIVED BUSINESS EVENTS
CREATE TABLE business_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES temporal_sessions(session_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,  -- 'enter_zone', 'exit_zone', 'wait_started', 'served', 'abandoned'
    timestamp TEXT NOT NULL,
    value REAL,                -- wait time or metric values
    zone_id TEXT
);

-- 5. VENUE MEMORY (Rolling baselines)
CREATE TABLE venue_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id TEXT DEFAULT 'default',
    metric_type TEXT,          -- 'visitor_count', 'avg_dwell', 'abandonment_rate'
    day_of_week INTEGER,
    hour_of_day INTEGER,
    value REAL,
    sample_count INTEGER DEFAULT 1,
    updated_at TEXT
);

-- 6. QUERY EXAMPLES (Feedback few-shots)
CREATE TABLE query_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id TEXT DEFAULT 'default',
    question TEXT,
    sql_used TEXT,
    plain_answer TEXT,
    rating INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_start ON temporal_sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_business_events_type ON business_events(event_type);
CREATE INDEX IF NOT EXISTS idx_raw_obs_time ON raw_observations(timestamp);
""")

conn.commit()
conn.close()
print("SQLite Database successfully refactored and re-initialized at", db_path)
