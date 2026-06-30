"""
memory_updater.py
-----------------
Self-learning memory engine. Runs after every pipeline job and:
  1. Reads the freshly populated wait_metrics for this run.
  2. Computes per-hour / per-day-of-week rolling stats.
  3. Upserts into venue_memory using exponential moving average
     (newer runs have more weight; old patterns naturally fade).
  4. Provides build_context_snapshot() for the query agent to inject
     learned baselines into LLM prompts.
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "customer_intel.db")
ALPHA = 0.3  # EMA weight for new observations (0=never learn, 1=forget everything)


def _connect(db_path: str = DB_PATH):
    return sqlite3.connect(db_path)


# ── Core update ────────────────────────────────────────────────────────────────

def run_update(venue_id: str = "default", db_path: str = DB_PATH):
    """
    Called after every pipeline job. Reads fresh temporal_sessions + business_events data
    and updates venue_memory with rolling averages.
    """
    db = _connect(db_path)
    cur = db.cursor()
    now = datetime.now(timezone.utc).isoformat()

    # ── 1. Per-hour stats (customers only) ─────────────────────────────────────
    cur.execute("""
        SELECT
            CAST(strftime('%H', ts.start_time) AS INTEGER) AS hour,
            CAST(strftime('%w', ts.start_time) AS INTEGER) AS dow,
            COUNT(DISTINCT ts.session_id) AS visitor_count,
            AVG(ts.duration_seconds) AS avg_dwell,
            ROUND(
                100.0 * COUNT(DISTINCT CASE WHEN be.event_type='abandoned' THEN ts.session_id END)
                / MAX(COUNT(DISTINCT ts.session_id), 1),
                1
            ) AS abandonment_rate
        FROM temporal_sessions ts
        LEFT JOIN staff_resolutions sr ON ts.session_id = sr.session_id
        LEFT JOIN business_events be ON ts.session_id = be.session_id
        WHERE sr.id IS NULL AND ts.duration_seconds > 3
        GROUP BY hour, dow
    """)
    rows = cur.fetchall()

    for hour, dow, visitor_count, avg_dwell, abandonment_rate in rows:
        _upsert_metric(db, cur, venue_id, "visitor_count", dow, hour,
                       visitor_count, now)
        if avg_dwell is not None:
            _upsert_metric(db, cur, venue_id, "avg_dwell", dow, hour,
                           avg_dwell, now)
        if abandonment_rate is not None:
            _upsert_metric(db, cur, venue_id, "abandonment_rate", dow, hour,
                           abandonment_rate, now)

    # ── 2. Global time-to-service (from 'served' events) ──────────────────────
    cur.execute("""
        SELECT AVG(be.value)
        FROM business_events be
        JOIN temporal_sessions ts ON be.session_id = ts.session_id
        LEFT JOIN staff_resolutions sr ON ts.session_id = sr.session_id
        WHERE be.event_type = 'served'
          AND sr.id IS NULL
          AND ts.duration_seconds > 3
    """)
    row = cur.fetchone()
    if row and row[0] is not None:
        _upsert_metric(db, cur, venue_id, "avg_time_to_service", -1, -1,
                       row[0], now)

    db.commit()
    db.close()
    print(f"  [MEMORY] Updated venue_memory for venue='{venue_id}' — "
          f"{len(rows)} hour/day slots processed.")


def _upsert_metric(db, cur, venue_id, metric_type, dow, hour, new_value, now):
    """
    Upserts a single metric using exponential moving average.
    If no prior record exists, inserts with sample_count=1.
    """
    cur.execute("""
        SELECT id, value, sample_count FROM venue_memory
        WHERE venue_id=? AND metric_type=? AND day_of_week=? AND hour_of_day=?
    """, (venue_id, metric_type, dow, hour))
    existing = cur.fetchone()

    if existing:
        rec_id, old_value, sample_count = existing
        # Exponential moving average
        updated_value = ALPHA * new_value + (1 - ALPHA) * old_value
        new_count = sample_count + 1
        cur.execute("""
            UPDATE venue_memory
            SET value=?, sample_count=?, updated_at=?
            WHERE id=?
        """, (round(updated_value, 3), new_count, now, rec_id))
    else:
        cur.execute("""
            INSERT INTO venue_memory
              (venue_id, metric_type, day_of_week, hour_of_day, value, sample_count, updated_at)
            VALUES (?,?,?,?,?,1,?)
        """, (venue_id, metric_type, dow, hour, round(new_value, 3), now))


# ── Context snapshot for LLM ───────────────────────────────────────────────────

def build_context_snapshot(
    venue_id: str = "default",
    db_path: str = DB_PATH,
    top_examples: int = 5
) -> str:
    """
    Returns a plain-text context block to inject into LLM system prompts.
    Includes:
      - Learned baselines for the current day-of-week / hour
      - Anomaly flags if today's live data deviates from baseline
      - Top-rated Q&A examples from this venue (few-shot)
    """
    db = _connect(db_path)
    cur = db.cursor()

    now = datetime.now(timezone.utc)
    current_dow = now.weekday()      # 0=Mon..6=Sun (Python convention)
    # SQLite strftime('%w') is 0=Sun..6=Sat, so convert
    sqlite_dow = (current_dow + 1) % 7
    current_hour = now.hour

    lines = ["=== VENUE MEMORY & LEARNED BASELINES ==="]

    # ── Baselines for current hour/day ────────────────────────────────────────
    cur.execute("""
        SELECT metric_type, value, sample_count
        FROM venue_memory
        WHERE venue_id=? AND day_of_week=? AND hour_of_day=?
    """, (venue_id, sqlite_dow, current_hour))
    baseline_rows = cur.fetchall()

    if baseline_rows:
        dow_name = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][sqlite_dow]
        lines.append(
            f"\nLearned baseline for {dow_name} {current_hour:02d}:00 "
            f"(from {baseline_rows[0][2]} historical runs):"
        )
        for metric_type, value, sample_count in baseline_rows:
            label = {
                "avg_dwell": f"  • Typical avg dwell: {round(value)}s "
                             f"({round(value/60, 1)} min)",
                "visitor_count": f"  • Typical visitor count: {round(value)}",
                "abandonment_rate": f"  • Typical abandonment rate: {round(value, 1)}%",
                "avg_time_to_service": f"  • Typical time-to-service: {round(value)}s",
            }.get(metric_type, f"  • {metric_type}: {round(value, 2)}")
            lines.append(label)
    else:
        lines.append(
            "\nNo baseline yet for this time slot — this appears to be an early run. "
            "Answer using live data only."
        )

    # ── Anomaly detection (compare live vs baseline) ──────────────────────────
    anomalies = _detect_anomalies(cur, venue_id, sqlite_dow, current_hour)
    if anomalies:
        lines.append("\n⚠️  ANOMALIES DETECTED vs learned baseline:")
        for a in anomalies:
            lines.append(f"  • {a}")

    # ── All-time summary baselines (global) ───────────────────────────────────
    cur.execute("""
        SELECT metric_type, AVG(value)
        FROM venue_memory
        WHERE venue_id=?
        GROUP BY metric_type
    """, (venue_id,))
    global_rows = cur.fetchall()
    if global_rows:
        lines.append("\nAll-time venue averages (across all hours/days):")
        for metric_type, avg_val in global_rows:
            label = {
                "avg_dwell": f"  • Overall avg dwell: {round(avg_val)}s",
                "visitor_count": f"  • Overall avg visitors/hour: {round(avg_val, 1)}",
                "abandonment_rate": f"  • Overall abandonment rate: {round(avg_val, 1)}%",
                "avg_time_to_service": f"  • Overall time-to-service: {round(avg_val)}s",
            }.get(metric_type, f"  • {metric_type}: {round(avg_val, 2)}")
            lines.append(label)

    # ── Few-shot Q&A examples from this venue ─────────────────────────────────
    cur.execute("""
        SELECT question, plain_answer
        FROM query_examples
        WHERE venue_id=? AND rating=1
        ORDER BY created_at DESC
        LIMIT ?
    """, (venue_id, top_examples))
    examples = cur.fetchall()

    if examples:
        lines.append(
            f"\n=== TOP {len(examples)} APPROVED Q&A EXAMPLES FOR THIS VENUE ==="
        )
        for i, (q, a) in enumerate(examples, 1):
            lines.append(f"\nExample {i}:")
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")

    db.close()
    return "\n".join(lines)


def _detect_anomalies(cur, venue_id, dow, hour) -> list:
    """
    Compares live temporal_sessions data against stored baselines.
    Returns a list of human-readable anomaly strings.
    """
    anomalies = []

    # Live stats for current hour (customers only)
    cur.execute("""
        SELECT
            COUNT(DISTINCT ts.session_id) as live_visitors,
            AVG(ts.duration_seconds) as live_dwell,
            ROUND(
                100.0 * COUNT(DISTINCT CASE WHEN be.event_type='abandoned' THEN ts.session_id END)
                / MAX(COUNT(DISTINCT ts.session_id), 1),
                1
            ) as live_abandon
        FROM temporal_sessions ts
        LEFT JOIN staff_resolutions sr ON ts.session_id = sr.session_id
        LEFT JOIN business_events be ON ts.session_id = be.session_id
        WHERE sr.id IS NULL AND ts.duration_seconds > 3
          AND CAST(strftime('%H', ts.start_time) AS INTEGER) = ?
    """, (hour,))
    live = cur.fetchone()
    if not live or live[0] == 0:
        return anomalies

    live_visitors, live_dwell, live_abandon = live

    # Compare each metric against baseline
    checks = [
        ("visitor_count", live_visitors, "visitors"),
        ("avg_dwell", live_dwell, "avg dwell (seconds)"),
        ("abandonment_rate", live_abandon, "abandonment rate (%)"),
    ]

    for metric_type, live_val, label in checks:
        if live_val is None:
            continue
        cur.execute("""
            SELECT value FROM venue_memory
            WHERE venue_id=? AND metric_type=? AND day_of_week=? AND hour_of_day=?
        """, (venue_id, metric_type, dow, hour))
        row = cur.fetchone()
        if not row:
            continue
        baseline = row[0]
        if baseline == 0:
            continue

        ratio = live_val / baseline
        if ratio >= 1.5:
            anomalies.append(
                f"{label.title()} is {round(ratio, 1)}x above baseline "
                f"({round(live_val, 1)} vs typical {round(baseline, 1)})"
            )
        elif ratio <= 0.5:
            anomalies.append(
                f"{label.title()} is {round(1/ratio, 1)}x below baseline "
                f"({round(live_val, 1)} vs typical {round(baseline, 1)})"
            )

    return anomalies


# ── Standalone usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running memory updater...")
    run_update()
    print("\n--- Context Snapshot ---")
    print(build_context_snapshot())
