import sqlite3
import json
from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def evaluate_run(db_path='db/customer_intel.db'):
    """
    Evaluates the quality of a pipeline run.
    Returns a dict with pass/fail flags, issues found, and recommendations.
    """
    db = sqlite3.connect(db_path)
    cur = db.cursor()

    issues = []
    recommendations = []
    auto_fixes = []

    # ── Check 1: Zero/near-zero dwell times ──────────────────────────────
    cur.execute("SELECT COUNT(*) FROM wait_metrics WHERE wait_seconds <= 2")
    garbage_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM wait_metrics")
    total_count = cur.fetchone()[0]

    if garbage_count > 0:
        pct = round((garbage_count / max(total_count, 1)) * 100, 1)
        issues.append(f"{garbage_count} records have dwell ≤2s ({pct}% of total) — likely false detections")
        auto_fixes.append("remove_garbage_dwells")

    # ── Check 2: Token fragmentation ─────────────────────────────────────
    cur.execute("SELECT COUNT(DISTINCT token_id) FROM persons")
    unique_tokens = cur.fetchone()[0]
    cur.execute("""
        SELECT CAST(strftime('%s', MAX(last_seen)) AS INTEGER) -
               CAST(strftime('%s', MIN(first_seen)) AS INTEGER)
        FROM persons
    """)
    video_duration = cur.fetchone()[0] or 0

    # Heuristic: more than 1 new person every 5 seconds is suspicious
    expected_max = max(2, video_duration / 5)
    if unique_tokens > expected_max and video_duration > 0:
        issues.append(
            f"Token fragmentation detected: {unique_tokens} unique tokens for {video_duration}s video "
            f"(expected max ~{int(expected_max)})"
        )
        recommendations.append("Lower Re-ID threshold from 0.65 to 0.60 and reprocess")

    # ── Check 3: 100% abandonment sanity check ───────────────────────────
    cur.execute("SELECT COUNT(*) FROM wait_metrics WHERE abandoned=0 AND wait_seconds > 2")
    served_count = cur.fetchone()[0]
    if served_count == 0 and total_count > 0:
        issues.append("100% abandonment rate — staff detection not configured or venue has no attended zones")
        recommendations.append("Define attended zones or tag staff members for accurate served/unattended split")

    # ── Check 4: Dwell time plausibility ─────────────────────────────────
    cur.execute("SELECT MAX(wait_seconds) FROM wait_metrics")
    max_dwell = cur.fetchone()[0] or 0
    if max_dwell > 7200:
        issues.append(f"Max dwell of {max_dwell}s ({round(max_dwell/3600,1)}h) is implausible — possible tracking error")
        auto_fixes.append("cap_extreme_dwells")

    # ── Check 5: Duplicate entries for same token ────────────────────────
    cur.execute("""
        SELECT token_id, COUNT(*) as cnt
        FROM wait_metrics
        GROUP BY token_id
        HAVING cnt > 1
    """)
    duplicates = cur.fetchall()
    if duplicates:
        issues.append(f"{len(duplicates)} tokens have duplicate metric entries")
        auto_fixes.append("deduplicate_metrics")

    # ── Auto-fix: remove garbage dwells ──────────────────────────────────
    if "remove_garbage_dwells" in auto_fixes:
        cur.execute("DELETE FROM wait_metrics WHERE wait_seconds <= 2")
        db.commit()
        print(f"  [FIX] Removed {garbage_count} garbage dwell records")

    # ── Auto-fix: cap extreme dwells ─────────────────────────────────────
    if "cap_extreme_dwells" in auto_fixes:
        cur.execute("UPDATE wait_metrics SET wait_seconds = 7200 WHERE wait_seconds > 7200")
        db.commit()
        print(f"  [FIX] Capped extreme dwell times at 2 hours")

    # ── Auto-fix: deduplicate ─────────────────────────────────────────────
    if "deduplicate_metrics" in auto_fixes:
        cur.execute("""
            DELETE FROM wait_metrics
            WHERE id NOT IN (
                SELECT MIN(id) FROM wait_metrics GROUP BY token_id
            )
        """)
        db.commit()
        print(f"  [FIX] Deduplicated metric entries")

    # ── Final clean stats ─────────────────────────────────────────────────
    cur.execute("""
        SELECT COUNT(*), ROUND(AVG(wait_seconds),1), ROUND(MAX(wait_seconds),1)
        FROM wait_metrics WHERE wait_seconds > 0
    """)
    final = cur.fetchone()
    db.close()

    quality_score = max(0, 100 - (len(issues) * 20))
    passed = quality_score >= 60

    result = {
        "passed": passed,
        "quality_score": quality_score,
        "issues_found": len(issues),
        "issues": issues,
        "recommendations": recommendations,
        "auto_fixes_applied": auto_fixes,
        "final_stats": {
            "clean_records": final[0],
            "avg_dwell_seconds": final[1],
            "max_dwell_seconds": final[2]
        }
    }

    return result


def get_ai_summary(eval_result):
    """Ask Groq to summarize the evaluation in plain English for the dashboard."""
    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a data quality analyst. Summarize this pipeline evaluation result in 2-3 plain English sentences for a business owner. Focus on data quality and what was automatically fixed. Be reassuring if issues were auto-fixed."},
            {"role": "user", "content": json.dumps(eval_result)}
        ],
        max_tokens=150,
        temperature=0.3
    )
    return msg.choices[0].message.content.strip()


if __name__ == "__main__":
    print("Running evaluator...")
    result = evaluate_run()
    print(json.dumps(result, indent=2))
    print("\nAI Summary:")
    print(get_ai_summary(result))
