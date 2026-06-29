import sqlite3, os

conn = sqlite3.connect('db/customer_intel.db')
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS persons (
    token_id     TEXT PRIMARY KEY,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT,
    camera_id    TEXT,
    abandoned    INTEGER DEFAULT 0,
    is_staff     INTEGER DEFAULT 0,
    staff_id     TEXT
);

CREATE TABLE IF NOT EXISTS wait_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id        TEXT,
    entry_time      TEXT,
    exit_time       TEXT,
    wait_seconds    REAL,
    time_to_service REAL,
    abandoned       INTEGER DEFAULT 0,
    date            TEXT
);
CREATE INDEX IF NOT EXISTS idx_date ON wait_metrics(date);
CREATE INDEX IF NOT EXISTS idx_wait ON wait_metrics(wait_seconds);
""")

conn.commit()
conn.close()
print("SQLite DB ready at db/customer_intel.db")
