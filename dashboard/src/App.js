import { useState, useEffect, useRef } from "react";
import axios from "axios";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

const PRESET_QUESTIONS = [
  "How many visitors came today?",
  "How many people were in the venue between 8pm and 9pm?",
  "What was the average wait time this hour?",
  "How many people left without being served today?",
  "Which hour had the most visitors?",
  "What was the busiest time period today?",
  "Who stayed the longest and for how long?",
  "How does the abandonment rate look today?",
  "How many visitors came in the last 30 minutes?",
  "What percentage of visitors were served today?",
];

function KPICard({ icon, label, value, detail, color = "#f59e0b" }) {
  return (
    <div style={{
      background: "#111827", border: "1px solid #1f2937",
      borderRadius: 16, padding: "24px 28px", flex: 1, minWidth: 180,
      borderTop: `3px solid ${color}`
    }}>
      <div style={{ fontSize: 22, marginBottom: 8 }}>{icon}</div>
      <div style={{ color: "#6b7280", fontSize: 12, fontWeight: 600,
        textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{label}</div>
      <div style={{ color: "#f9fafb", fontSize: 32, fontWeight: 800,
        lineHeight: 1, marginBottom: 6 }}>{value}</div>
      {detail && <div style={{ color: "#4b5563", fontSize: 12 }}>{detail}</div>}
    </div>
  );
}

function dwellLabel(s) {
  if (!s) return "—";
  if (s < 60) return `${parseFloat(s).toFixed(0)}s`;
  return `${(s / 60).toFixed(1)} min`;
}

function formatTime(iso) {
  if (!iso) return "—";
  return iso.slice(11, 19) + " UTC";
}

function UploadScreen({ onProcessingStart }) {
  const [dragOver, setDragOver] = useState(false);
  const [urlInput, setUrlInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef();

  const handleFile = async (file) => {
    if (!file) return;
    setUploading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await axios.post(`${API}/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      onProcessingStart(r.data.job_id);
    } catch (e) {
      setError("Upload failed. Make sure the API is running.");
      setUploading(false);
    }
  };

  const handleUrl = async () => {
    if (!urlInput.trim()) return;
    setUploading(true);
    setError("");
    try {
      const r = await axios.post(`${API}/upload/url`, { url: urlInput });
      if (r.data.error) { setError(r.data.error); setUploading(false); return; }
      onProcessingStart(r.data.job_id);
    } catch (e) {
      setError("Failed to process URL.");
      setUploading(false);
    }
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#0a0f1a",
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", padding: 40, fontFamily: "'Inter', sans-serif",
    }}>
      <div style={{ color: "#f59e0b", fontSize: 11, fontWeight: 700,
        letterSpacing: 3, textTransform: "uppercase", marginBottom: 12 }}>
        Powered by Computer Vision + AI
      </div>
      <h1 style={{ color: "#f9fafb", fontSize: 36, fontWeight: 800,
        margin: "0 0 8px", textAlign: "center" }}>
        Customer Intelligence Platform
      </h1>
      <p style={{ color: "#4b5563", fontSize: 15, marginBottom: 48,
        textAlign: "center", maxWidth: 500 }}>
        Upload your venue footage and get instant insights on visitor behaviour,
        dwell time, and service efficiency.
      </p>

      <div style={{ width: "100%", maxWidth: 560 }}>
        {/* Drag & Drop */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
          onClick={() => fileRef.current.click()}
          style={{
            border: `2px dashed ${dragOver ? "#f59e0b" : "#374151"}`,
            borderRadius: 16, padding: "48px 32px", textAlign: "center",
            cursor: "pointer", marginBottom: 24,
            background: dragOver ? "#1a1500" : "#111827", transition: "all 0.2s",
          }}
        >
          <div style={{ fontSize: 40, marginBottom: 12 }}>🎥</div>
          <div style={{ color: "#f9fafb", fontWeight: 600, fontSize: 16, marginBottom: 6 }}>
            Drop your video here
          </div>
          <div style={{ color: "#4b5563", fontSize: 13 }}>
            MP4, MOV, AVI supported · or click to browse
          </div>
          <input ref={fileRef} type="file" accept="video/*"
            style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
        </div>

        {/* Divider */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
          <div style={{ flex: 1, height: 1, background: "#1f2937" }} />
          <span style={{ color: "#4b5563", fontSize: 12, fontWeight: 600,
            textTransform: "uppercase", letterSpacing: 1 }}>or paste a YouTube URL</span>
          <div style={{ flex: 1, height: 1, background: "#1f2937" }} />
        </div>

        {/* URL input */}
        <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
          <input
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleUrl()}
            placeholder="https://www.youtube.com/watch?v=..."
            style={{
              flex: 1, background: "#111827", border: "1px solid #374151",
              borderRadius: 10, padding: "12px 18px", color: "#f9fafb",
              fontSize: 14, outline: "none"
            }}
          />
          <button onClick={handleUrl} disabled={uploading} style={{
            background: "#f59e0b", color: "#0a0f1a", border: "none",
            borderRadius: 10, padding: "12px 24px", fontWeight: 800,
            fontSize: 14, cursor: "pointer"
          }}>
            Analyse URL
          </button>
        </div>

        {error && (
          <div style={{ background: "#1c0a00", border: "1px solid #7c2d12",
            borderRadius: 10, padding: "12px 16px", color: "#f97316",
            fontSize: 13, marginBottom: 16 }}>{error}</div>
        )}
        {uploading && (
          <div style={{ textAlign: "center", color: "#6b7280", fontSize: 13 }}>
            Uploading video...
          </div>
        )}

        {/* Feature list */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr",
          gap: 12, marginTop: 40 }}>
          {[
            ["👥", "Person Detection", "Identifies every unique visitor"],
            ["🔄", "Cross-frame Tracking", "Follows people across camera angles"],
            ["⏱", "Dwell Time Analysis", "Measures how long each visitor stays"],
            ["🤖", "AI Business Insights", "Ask questions in plain English"],
          ].map(([icon, title, desc]) => (
            <div key={title} style={{ background: "#111827",
              border: "1px solid #1f2937", borderRadius: 12, padding: 16 }}>
              <div style={{ fontSize: 20, marginBottom: 8 }}>{icon}</div>
              <div style={{ color: "#f9fafb", fontWeight: 600,
                fontSize: 13, marginBottom: 4 }}>{title}</div>
              <div style={{ color: "#4b5563", fontSize: 12 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ProcessingScreen({ jobId, onComplete }) {
  const [job, setJob] = useState({ status: "queued", progress: 0, stage: "Queued" });

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const r = await axios.get(`${API}/job/${jobId}`);
        setJob(r.data);
        if (r.data.status === "done" || r.data.status === "error") {
          clearInterval(interval);
          if (r.data.status === "done") setTimeout(onComplete, 800);
        }
      } catch (e) {}
    }, 1500);
    return () => clearInterval(interval);
  }, [jobId, onComplete]);

  const stages = [
    "Analysing video characteristics...",
    "Detecting people...",
    "Building identity profiles...",
    "Analysing behaviour...",
    "Evaluating data quality...",
    "Generating insights...",
    "Complete",
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#0a0f1a",
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", padding: 40, fontFamily: "'Inter', sans-serif" }}>
      <div style={{ width: "100%", maxWidth: 480, textAlign: "center" }}>
        <div style={{ fontSize: 48, marginBottom: 24 }}>
          {job.status === "error" ? "❌" : job.status === "done" ? "✅" : "⚙️"}
        </div>
        <h2 style={{ color: "#f9fafb", fontSize: 24, fontWeight: 800, marginBottom: 8 }}>
          {job.status === "error" ? "Processing Failed" :
           job.status === "done" ? "Analysis Complete!" : "Analysing Your Video"}
        </h2>
        <p style={{ color: "#6b7280", fontSize: 14, marginBottom: 40 }}>{job.stage}</p>

        {/* Progress bar */}
        <div style={{ background: "#1f2937", borderRadius: 8, height: 8,
          marginBottom: 32, overflow: "hidden" }}>
          <div style={{
            height: "100%", borderRadius: 8,
            background: job.status === "error" ? "#ef4444" : "#f59e0b",
            width: `${job.progress || 0}%`, transition: "width 0.5s ease"
          }} />
        </div>

        {/* Stage checklist */}
        <div style={{ textAlign: "left" }}>
          {stages.map((s, i) => {
            const stageProgress = (i / (stages.length - 1)) * 100;
            const done = (job.progress || 0) > stageProgress;
            const active = job.stage === s;
            return (
              <div key={s} style={{ display: "flex", alignItems: "center",
                gap: 12, padding: "8px 0", borderBottom: "1px solid #1f2937" }}>
                <div style={{
                  width: 20, height: 20, borderRadius: "50%",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 11, fontWeight: 700, flexShrink: 0,
                  background: done ? "#f59e0b" : active ? "#1f2937" : "#111827",
                  color: done ? "#0a0f1a" : active ? "#f59e0b" : "#374151",
                  border: active ? "2px solid #f59e0b" : "2px solid transparent",
                }}>
                  {done ? "✓" : i + 1}
                </div>
                <span style={{
                  color: done ? "#f9fafb" : active ? "#f59e0b" : "#374151",
                  fontSize: 13, fontWeight: active ? 600 : 400
                }}>{s}</span>
              </div>
            );
          })}
        </div>

        {job.status === "error" && (
          <div style={{ marginTop: 24, color: "#f97316", fontSize: 13 }}>
            {job.stage}
          </div>
        )}
      </div>
    </div>
  );
}

function DashboardScreen({ onReset }) {
  const [summary, setSummary] = useState(null);
  const [persons, setPersons] = useState([]);
  const [hourly, setHourly] = useState([]);
  const [biq, setBiq] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchData = () => {
    axios.get(`${API}/metrics/summary`).then(r => {
      setSummary(r.data);
      setLastUpdated(new Date().toLocaleTimeString());
    }).catch(() => {});
    axios.get(`${API}/metrics/persons`).then(r => setPersons(r.data)).catch(() => {});
    axios.get(`${API}/metrics/hourly`).then(r => setHourly(r.data)).catch(() => {});
    axios.get(`${API}/metrics/business_iq`).then(r => setBiq(r.data)).catch(() => {});
  };

  useEffect(() => { fetchData(); }, []); // eslint-disable-line

  const askQuestion = async (q) => {
    const query = q || question;
    if (!query.trim()) return;
    setQuestion(query);
    setLoading(true);
    setAnswer(null);
    try {
      const r = await axios.post(`${API}/ask`, { question: query });
      setAnswer(r.data);
    } catch (e) {
      setAnswer({ plain_answer: "Could not reach the analytics engine." });
    }
    setLoading(false);
  };

  const chartData = persons.filter(p => p.dwell_seconds > 0);
  const maxDwell = Math.max(...chartData.map(d => d.dwell_seconds), 1);

  return (
    <div style={{ minHeight: "100vh", background: "#0a0f1a",
      color: "#f9fafb", fontFamily: "'Inter', sans-serif",
      padding: "36px 40px", maxWidth: 1100, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between",
        alignItems: "flex-start", marginBottom: 36 }}>
        <div>
          <div style={{ color: "#f59e0b", fontSize: 11, fontWeight: 700,
            letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>
            Venue Analytics
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 800, margin: 0 }}>Customer Intelligence</h1>
          <p style={{ color: "#4b5563", fontSize: 13, margin: "6px 0 0" }}>
            AI-powered insights from your venue footage
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span style={{ color: "#4b5563", fontSize: 12 }}>Updated {lastUpdated}</span>
          <button onClick={fetchData} style={{
            background: "transparent", border: "1px solid #374151",
            color: "#9ca3af", borderRadius: 8, padding: "6px 14px",
            fontSize: 12, cursor: "pointer" }}>↻</button>
          <button onClick={onReset} style={{
            background: "#1f2937", border: "1px solid #374151",
            color: "#9ca3af", borderRadius: 8, padding: "6px 14px",
            fontSize: 12, cursor: "pointer" }}>+ New Video</button>
        </div>
      </div>

      {/* KPIs */}
      {summary && (
        <div style={{ display: "flex", gap: 16, marginBottom: 28, flexWrap: "wrap" }}>
          <KPICard icon="👥" label="Total Visitors" color="#f59e0b"
            value={summary.total_visitors} detail="unique persons identified" />
          <KPICard icon="⏱" label="Avg Time in Venue" color="#3b82f6"
            value={dwellLabel(summary.avg_dwell_seconds)} detail="from entry to exit" />
          <KPICard icon="📈" label="Longest Stay" color="#8b5cf6"
            value={dwellLabel(summary.max_dwell_seconds)} detail="maximum dwell recorded" />
          <KPICard icon="🚪" label="Left Unattended"
            color={summary.abandonment_rate_pct > 50 ? "#ef4444" : "#10b981"}
            value={`${summary.abandonment_rate_pct}%`}
            detail={`${summary.abandoned_count} of ${summary.total_visitors} visitors`} />
        </div>
      )}

      {/* Business IQ */}
      {biq && (
        <div style={{ background: "#111827", border: "1px solid #1f2937",
          borderRadius: 16, padding: "28px", marginBottom: 28,
          display: "flex", gap: 32, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ textAlign: "center", flexShrink: 0 }}>
            <div style={{
              width: 120, height: 120, borderRadius: "50%",
              border: `6px solid ${biq.color}`,
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              background: "#0a0f1a"
            }}>
              <div style={{ color: biq.color, fontSize: 36, fontWeight: 900, lineHeight: 1 }}>{biq.score}</div>
              <div style={{ color: biq.color, fontSize: 22, fontWeight: 800 }}>{biq.grade}</div>
            </div>
            <div style={{ color: "#6b7280", fontSize: 11, fontWeight: 600,
              textTransform: "uppercase", letterSpacing: 1, marginTop: 10 }}>Business IQ</div>
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ color: "#f9fafb", fontWeight: 700, fontSize: 16, marginBottom: 4 }}>
              Venue Performance Score
            </div>
            <div style={{ color: "#4b5563", fontSize: 13, marginBottom: 20 }}>
              Composite score based on service rate, dwell time, and abandonment
            </div>
            {[
              ["Service Rate", biq.breakdown.service_score, "#10b981"],
              ["Dwell Quality", biq.breakdown.dwell_score, "#3b82f6"],
              ["Attendance Score", biq.breakdown.abandonment_score, "#f59e0b"],
            ].map(([label, score, color]) => (
              <div key={label} style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between",
                  fontSize: 12, marginBottom: 4 }}>
                  <span style={{ color: "#9ca3af" }}>{label}</span>
                  <span style={{ color: "#f9fafb", fontWeight: 600 }}>{score}/100</span>
                </div>
                <div style={{ background: "#1f2937", borderRadius: 4, height: 6 }}>
                  <div style={{ width: `${score}%`, height: "100%",
                    background: color, borderRadius: 4, transition: "width 0.8s ease" }} />
                </div>
              </div>
            ))}
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ color: "#6b7280", fontSize: 11, fontWeight: 600,
              textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
              What This Means
            </div>
            {biq.insights && biq.insights.map((ins, i) => (
              <div key={i} style={{ display: "flex", gap: 10, marginBottom: 10, fontSize: 13 }}>
                <span style={{ color: biq.color, flexShrink: 0 }}>→</span>
                <span style={{ color: "#9ca3af" }}>{ins}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Smart Insights */}
      {summary && (
        <div style={{ background: "#111827", border: "1px solid #1f2937",
          borderRadius: 16, padding: "20px 24px", marginBottom: 28 }}>
          <div style={{ color: "#6b7280", fontSize: 12, fontWeight: 600,
            textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>
            Smart Insights
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {[
              summary.abandonment_rate_pct === 100
                ? { text: "⚠️ All visitors left unattended — staff response needed", type: "warning" }
                : summary.abandonment_rate_pct > 50
                ? { text: `⚠️ ${summary.abandonment_rate_pct}% abandonment rate — high`, type: "warning" }
                : { text: `✅ ${100 - summary.abandonment_rate_pct}% of visitors were attended`, type: "good" },
              summary.avg_dwell_seconds > 300
                ? { text: "⏱ Avg wait over 5 min — consider more floor staff", type: "warning" }
                : { text: `⏱ Avg dwell ${dwellLabel(summary.avg_dwell_seconds)} — within normal range`, type: "good" },
              { text: `👥 ${summary.total_visitors} unique visitor${summary.total_visitors !== 1 ? "s" : ""} tracked`, type: "info" },
            ].map((ins, i) => (
              <span key={i} style={{
                background: ins.type === "warning" ? "#451a03" : ins.type === "good" ? "#052e16" : "#0c1a3a",
                color: ins.type === "warning" ? "#fb923c" : ins.type === "good" ? "#4ade80" : "#60a5fa",
                padding: "6px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600
              }}>{ins.text}</span>
            ))}
          </div>
        </div>
      )}

      {/* Dwell Chart */}
      <div style={{ background: "#111827", border: "1px solid #1f2937",
        borderRadius: 16, padding: "24px 28px", marginBottom: 28 }}>
        <div style={{ color: "#6b7280", fontSize: 12, fontWeight: 600,
          textTransform: "uppercase", letterSpacing: 1, marginBottom: 20 }}>
          Time Spent in Venue per Visitor
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {chartData.map((p, i) => (
            <div key={p.token_id}>
              <div style={{ display: "flex", justifyContent: "space-between",
                marginBottom: 6, fontSize: 13 }}>
                <span style={{ color: "#9ca3af", fontWeight: 600 }}>Visitor {i + 1}</span>
                <span style={{ color: "#f9fafb", fontWeight: 700 }}>{dwellLabel(p.dwell_seconds)}</span>
              </div>
              <div style={{ background: "#1f2937", borderRadius: 6, height: 10, overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  width: `${Math.min((p.dwell_seconds / (maxDwell * 1.2)) * 100, 100)}%`,
                  background: p.abandoned ? "#f59e0b" : "#10b981",
                  borderRadius: 6, transition: "width 0.6s ease"
                }} />
              </div>
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: 20, marginTop: 16 }}>
          {[["#10b981", "Served by staff"], ["#f59e0b", "Left unattended"]].map(([c, l]) => (
            <div key={l} style={{ display: "flex", alignItems: "center",
              gap: 6, fontSize: 12, color: "#6b7280" }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />
              {l}
            </div>
          ))}
        </div>
      </div>

      {/* Hourly Traffic */}
      {hourly.length > 0 && (
        <div style={{ background: "#111827", border: "1px solid #1f2937",
          borderRadius: 16, padding: "24px 28px", marginBottom: 28 }}>
          <div style={{ color: "#6b7280", fontSize: 12, fontWeight: 600,
            textTransform: "uppercase", letterSpacing: 1, marginBottom: 20 }}>
            Visitor Traffic by Hour
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {hourly.map((h) => (
              <div key={h.hour}>
                <div style={{ display: "flex", justifyContent: "space-between",
                  marginBottom: 5, fontSize: 13 }}>
                  <span style={{ color: "#9ca3af", fontWeight: 600 }}>
                    {h.hour}:00 – {h.hour + 1}:00 UTC
                  </span>
                  <span style={{ color: "#f9fafb", fontWeight: 700 }}>
                    {h.visitors} visitor{h.visitors !== 1 ? "s" : ""} · avg {dwellLabel(h.avg_dwell_seconds)}
                  </span>
                </div>
                <div style={{ background: "#1f2937", borderRadius: 6, height: 10, overflow: "hidden" }}>
                  <div style={{
                    height: "100%",
                    width: `${Math.min((h.visitors / Math.max(...hourly.map(x => x.visitors))) * 100, 100)}%`,
                    background: "#3b82f6", borderRadius: 6, transition: "width 0.6s ease"
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Visitor Log */}
      <div style={{ background: "#111827", border: "1px solid #1f2937",
        borderRadius: 16, padding: "24px 28px", marginBottom: 28 }}>
        <div style={{ color: "#6b7280", fontSize: 12, fontWeight: 600,
          textTransform: "uppercase", letterSpacing: 1, marginBottom: 20 }}>
          Visitor Log
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              {["Visitor", "Arrived", "Camera", "Time in Venue", "Outcome"].map(h => (
                <th key={h} style={{ textAlign: "left", padding: "8px 16px",
                  color: "#4b5563", fontWeight: 600, fontSize: 11,
                  textTransform: "uppercase", borderBottom: "1px solid #1f2937" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {persons.map((p, i) => (
              <tr key={p.token_id} style={{ borderBottom: "1px solid #111827" }}>
                <td style={{ padding: "14px 16px", color: "#9ca3af", fontWeight: 600 }}>
                  Visitor {i + 1}
                </td>
                <td style={{ padding: "14px 16px", color: "#6b7280" }}>{formatTime(p.entered)}</td>
                <td style={{ padding: "14px 16px", color: "#6b7280" }}>
                  {p.camera === "cam_01" ? "Entrance Camera" : p.camera}
                </td>
                <td style={{ padding: "14px 16px", color: "#f9fafb", fontWeight: 700 }}>
                  {dwellLabel(p.dwell_seconds)}
                </td>
                <td style={{ padding: "14px 16px" }}>
                  <span style={{
                    background: p.abandoned ? "#1c0a00" : "#052e16",
                    color: p.abandoned ? "#f97316" : "#4ade80",
                    padding: "4px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600
                  }}>
                    {p.abandoned ? "Left unattended" : "✓ Served"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* NL Query */}
      <div style={{ background: "#111827", border: "1px solid #1f2937",
        borderRadius: 16, padding: "24px 28px" }}>
        <div style={{ color: "#6b7280", fontSize: 12, fontWeight: 600,
          textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>
          Ask About Your Venue
        </div>
        <p style={{ color: "#4b5563", fontSize: 13, margin: "0 0 16px" }}>
          Ask anything in plain English — powered by AI, no technical knowledge needed.
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
          {PRESET_QUESTIONS.map(q => (
            <button key={q} onClick={() => askQuestion(q)} style={{
              background: "#1f2937", border: "1px solid #374151",
              color: "#9ca3af", borderRadius: 20, padding: "7px 16px",
              fontSize: 12, cursor: "pointer"
            }}>{q}</button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && askQuestion()}
            placeholder="Ask your own question..."
            style={{
              flex: 1, background: "#0a0f1a", border: "1px solid #374151",
              borderRadius: 10, padding: "12px 18px", color: "#f9fafb",
              fontSize: 14, outline: "none"
            }}
          />
          <button onClick={() => askQuestion()} disabled={loading} style={{
            background: "#f59e0b", color: "#0a0f1a", border: "none",
            borderRadius: 10, padding: "12px 28px", fontWeight: 800,
            fontSize: 14, cursor: "pointer", opacity: loading ? 0.7 : 1
          }}>
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>
        {answer && (
          <div style={{ background: "#0a0f1a", borderRadius: 12,
            padding: "20px 24px", border: "1px solid #1f2937" }}>
            <div style={{ color: "#f59e0b", fontWeight: 700, fontSize: 13,
              marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Answer</div>
            <div style={{ color: "#f9fafb", fontSize: 15, lineHeight: 1.8 }}>
              {answer.plain_answer}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [screen, setScreen] = useState("upload");
  const [jobId, setJobId] = useState(null);

  return screen === "upload" ? (
    <UploadScreen onProcessingStart={(id) => { setJobId(id); setScreen("processing"); }} />
  ) : screen === "processing" ? (
    <ProcessingScreen jobId={jobId} onComplete={() => setScreen("dashboard")} />
  ) : (
    <DashboardScreen onReset={() => { setJobId(null); setScreen("upload"); }} />
  );
}
