"""
Streamlit Presentation Layer — Sistem Penilaian Risiko Banjir Kota Padang
Single-page design with live progress tracking.
Core pipeline (src/) adalah IMMUTABLE. File ini hanya presentation layer.
"""

import streamlit as st
import pandas as pd
import sys
import os
import re
import time
import threading
import queue

# Workspace root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import predict_for_date
from config.settings import CLASS_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Penilaian Risiko Banjir Kota Padang",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap');

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

html, body, .stApp, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #f8f8f7 !important;
    color: #111 !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
header { visibility: hidden !important; }
section[data-testid="stSidebar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* Remove default block padding and force centering with max-width */
.block-container {
    max-width: 1100px !important;
    padding-top: 58px !important; /* height of navbar */
    padding-bottom: 80px !important;
    padding-left: 64px !important;
    padding-right: 64px !important;
    margin: 0 auto !important;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 24px !important;
        padding-right: 24px !important;
    }
}

/* ── Fixed Navbar ── */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    height: 58px;
    background: rgba(248, 248, 247, 0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid #e4e4e0;
}
.navbar-inner {
    max-width: 1100px;
    height: 100%;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 64px;
}
@media (max-width: 768px) {
    .navbar-inner {
        padding: 0 24px;
    }
}
.nav-brand {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #111;
    text-decoration: none;
}
.nav-links {
    display: flex;
    align-items: center;
    gap: 40px;
}
.nav-links a {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #666;
    text-decoration: none;
    transition: color 0.2s;
}
.nav-links a:hover { color: #111; }
.nav-spacer { height: 58px; }

/* ── Page Sections ── */
.page-section {
    width: 100%;
    padding: 60px 0 40px;
}
.page-section-inner {
    width: 100%;
}
.section-divider {
    width: 100%;
    border: none;
    border-top: 1px solid #e4e4e0;
    margin: 0;
}

/* ── Hero ── */
.hero-wrap { min-height: calc(100vh - 58px); display: flex; flex-direction: column; justify-content: center; }
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 28px;
}
.hero-title {
    font-size: clamp(3rem, 6.5vw, 5.5rem);
    font-weight: 800;
    line-height: 1.04;
    letter-spacing: -0.04em;
    color: #0d0d0d;
    margin-bottom: 32px;
}
.hero-title em { font-style: normal; color: #666; }
.hero-body {
    font-size: 1rem;
    line-height: 1.75;
    color: #555;
    max-width: 520px;
    margin-bottom: 52px;
    font-weight: 400;
}
.hero-cta {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: #0d0d0d;
    color: #fff !important;
    text-decoration: none !important;
    padding: 15px 36px;
    border-radius: 3px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transition: background 0.2s, transform 0.15s;
}
.hero-cta:hover { background: #2a2a2a; transform: translateY(-1px); }
.hero-stats {
    display: flex;
    gap: 0;
    border-top: 1px solid #e4e4e0;
    margin-top: 80px;
    padding-top: 0;
}
.hero-stat {
    flex: 1;
    padding: 28px 0;
    border-right: 1px solid #e4e4e0;
}
.hero-stat:last-child { border-right: none; }
.hero-stat-val {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0d0d0d;
    margin-bottom: 4px;
}
.hero-stat-lbl {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aaa;
}

/* ── Section Typography ── */
.s-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 14px;
}
.s-title {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #0d0d0d;
    margin-bottom: 14px;
    line-height: 1.1;
}
.s-body {
    font-size: 0.9rem;
    line-height: 1.75;
    color: #666;
    max-width: 520px;
    margin-bottom: 48px;
}

/* ── Form Controls ── */
.stDateInput label, .stNumberInput label, .stTextInput label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    margin-bottom: 6px !important;
}
div[data-testid="stDateInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #ddd !important;
    border-radius: 3px !important;
}
div[data-testid="stDateInput"] input,
div[data-testid="stTextInput"] input {
    color: #111 !important;
}

/* ── Buttons ── */
div.stButton > button,
div.stDownloadButton > button {
    background: #0d0d0d !important;
    color: #f8f8f7 !important;
    border: 1px solid #0d0d0d !important;
    border-radius: 3px !important;
    padding: 13px 32px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: background 0.2s, border-color 0.2s !important;
    cursor: pointer !important;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #2a2a2a !important;
    border-color: #2a2a2a !important;
    color: #ffffff !important;
}
div.stButton > button:focus,
div.stDownloadButton > button:focus,
div.stButton > button:active,
div.stDownloadButton > button:active {
    background: #2a2a2a !important;
    border-color: #2a2a2a !important;
    color: #ffffff !important;
    outline: none !important;
    box-shadow: none !important;
}
div.stButton > button:disabled,
div.stDownloadButton > button:disabled {
    background: #e4e4e0 !important;
    color: #777 !important;
    border-color: #e4e4e0 !important;
    cursor: not-allowed !important;
}

/* ── Expanders ── */
div[data-testid="stExpander"] details summary {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #e4e4e0 !important;
    border-radius: 3px !important;
    transition: background-color 0.15s, color 0.15s !important;
}
div[data-testid="stExpander"] details summary:hover {
    background-color: #f5f5f4 !important;
    color: #111111 !important;
}
div[data-testid="stExpander"] details summary:focus,
div[data-testid="stExpander"] details summary:active {
    background-color: #f5f5f4 !important;
    color: #111111 !important;
    outline: none !important;
}
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span,
div[data-testid="stExpander"] details summary svg {
    color: inherit !important;
    fill: currentColor !important;
}

/* ── Stage Tracker ── */
.stage-tracker {
    border: 1px solid #e4e4e0;
    border-radius: 6px;
    background: #fff;
    overflow: hidden;
    margin: 28px 0;
}
.stage-row {
    display: flex;
    align-items: center;
    padding: 13px 20px;
    border-bottom: 1px solid #f0f0ee;
    gap: 14px;
    transition: background 0.1s;
}
.stage-row:last-child { border-bottom: none; }
.stage-row.running { background: #fafaf9; }
.stage-row.done { }
.stage-row.pending { opacity: 0.45; }
.stage-icon {
    width: 22px;
    height: 22px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.s-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ccc;
}
.s-spinner {
    width: 14px;
    height: 14px;
    border: 2px solid #e4e4e0;
    border-top-color: #111;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
}
.s-check { font-size: 0.8rem; color: #111; font-weight: 700; }
.s-cross { font-size: 0.8rem; color: #c00; font-weight: 700; }
@keyframes spin { to { transform: rotate(360deg); } }
.stage-name {
    font-size: 0.825rem;
    font-weight: 500;
    color: #333;
    flex: 1;
    letter-spacing: 0.01em;
}
.stage-time {
    font-size: 0.72rem;
    color: #bbb;
    font-variant-numeric: tabular-nums;
}

/* ── Risk Display ── */
.result-wrap {
    border-top: 1px solid #e4e4e0;
    margin-top: 40px;
    padding-top: 40px;
}
.result-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 10px;
}
.result-class {
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 40px;
}
.rc-rendah { color: #1a1a1a; }
.rc-sedang { color: #92400e; }
.rc-tinggi { color: #991b1b; }
.rc-sangat-tinggi { color: #6b21a8; }

/* ── Probability Grid ── */
.prob-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 48px;
}
.prob-card {
    border: 1px solid #e4e4e0;
    border-radius: 4px;
    padding: 18px 16px;
    background: #fff;
    transition: border-color 0.2s;
}
.prob-card.active { border-color: #0d0d0d; }
.prob-card-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 10px;
}
.prob-card-val {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0d0d0d;
    margin-bottom: 10px;
}
.pbar-bg { height: 2px; background: #eee; border-radius: 1px; }
.pbar-fill { height: 2px; background: #0d0d0d; border-radius: 1px; }

/* ── Indices Grid ── */
.indices-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 16px;
}
.indices-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 40px;
}
.idx-card {
    border: 1px solid #e4e4e0;
    border-radius: 4px;
    padding: 14px 16px;
    background: #fff;
}
.idx-lbl {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 6px;
}
.idx-val {
    font-size: 1rem;
    font-weight: 700;
    color: #0d0d0d;
    letter-spacing: -0.01em;
}
.idx-na {
    font-size: 0.78rem;
    color: #ccc;
    font-style: italic;
}

/* ── Riwayat ── */
.empty-state {
    border: 1px dashed #ddd;
    border-radius: 6px;
    padding: 64px 40px;
    text-align: center;
    color: #bbb;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}

/* ── About ── */
.about-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 48px 64px;
}
.about-block {}
.about-block-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #555;
    padding-bottom: 12px;
    border-bottom: 1px solid #e4e4e0;
    margin-bottom: 16px;
}
.about-list { list-style: none; padding: 0; }
.about-list li {
    font-size: 0.85rem;
    line-height: 1.65;
    color: #555;
    padding: 7px 0;
    border-bottom: 1px solid #f5f5f3;
    display: flex;
    gap: 12px;
    align-items: baseline;
}
.about-list li::before {
    content: '';
    flex-shrink: 0;
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: #ccc;
    margin-top: 9px;
}

/* ── Misc ── */
.stSpinner > div > div { border-top-color: #111 !important; }
.stAlert { border-radius: 4px !important; font-size: 0.875rem !important; }
div[data-testid="stDataFrame"] { border: 1px solid #e4e4e0 !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "running": False,
    "result": None,
    "target_date": None,
    "history": [],
    "csv_path": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_STAGES = [
    {"key": "[1/7]", "label": "Mengambil data cuaca permukaan"},
    {"key": "[2/7]", "label": "Mengambil data atmosfer"},
    {"key": "[3/7]", "label": "Menghitung indeks atmosfer"},
    {"key": "[4/7]", "label": "Menggabungkan data"},
    {"key": "[5/7]", "label": "Menyiapkan data untuk prediksi"},
    {"key": "[6/7]", "label": "Memeriksa kelengkapan data 30 hari"},
    {"key": "[7/7]", "label": "Menjalankan model prediksi"},
]

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE STDOUT CAPTURE
# Menggunakan thread-aware writer: hanya menulis ke queue untuk thread
# inference, thread lain tetap menulis ke stdout asli.
# Tidak menyentuh pipeline sama sekali — hanya membaca stdout-nya.
# ─────────────────────────────────────────────────────────────────────────────
class ThreadAwareWriter:
    """Route writes from inference thread to queue; all others → original."""

    def __init__(self, original, log_queue, inference_thread_id):
        self._orig = original
        self._queue = log_queue
        self._tid = inference_thread_id

    def write(self, text):
        if threading.current_thread().ident == self._tid:
            if text:
                self._queue.put(text)
        else:
            try:
                self._orig.write(text)
            except Exception:
                pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass


def _inference_worker(target_date_str, result_ref, log_queue):
    """Run predict_for_date in a background thread."""
    try:
        res = predict_for_date(target_date_str, verbose=True, use_cache=True)
        result_ref["result"] = res
    except Exception as exc:
        result_ref["result"] = {
            "status": "REJECTED",
            "target_date": target_date_str,
            "reasons": [str(exc)],
        }
    finally:
        log_queue.put(None)  # sentinel — signals completion


# ─────────────────────────────────────────────────────────────────────────────
# STAGE TRACKER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _parse_stage(line: str, statuses: list):
    """Parse one pipeline print line; update statuses in-place."""
    for i, stage in enumerate(PIPELINE_STAGES):
        if stage["key"] not in line:
            continue
        if "OK" in line or "FAILED" in line:
            m = re.search(r"\((\d+\.\d+)s\)", line)
            statuses[i]["elapsed"] = m.group(1) if m else None
            statuses[i]["status"] = "done" if "OK" in line else "failed"
            # Advance next stage to 'running'
            if statuses[i]["status"] == "done" and i + 1 < len(PIPELINE_STAGES):
                if statuses[i + 1]["status"] == "pending":
                    statuses[i + 1]["status"] = "running"
        else:
            if statuses[i]["status"] == "pending":
                statuses[i]["status"] = "running"
        break


def _render_tracker(statuses: list) -> str:
    rows = []
    for i, stage in enumerate(PIPELINE_STAGES):
        s = statuses[i]
        st_cls = s["status"]
        if st_cls == "pending":
            icon = '<div class="s-dot"></div>'
        elif st_cls == "running":
            icon = '<div class="s-spinner"></div>'
        elif st_cls == "done":
            icon = '<span class="s-check">✓</span>'
        else:
            icon = '<span class="s-cross">✕</span>'
        elapsed = f'<span class="stage-time">{s["elapsed"]}s</span>' if s.get("elapsed") else ""
        rows.append(
            f'<div class="stage-row {st_cls}">'
            f'<div class="stage-icon">{icon}</div>'
            f'<span class="stage-name">{stage["label"]}</span>'
            f'{elapsed}'
            f'</div>'
        )
    return f'<div class="stage-tracker">{"".join(rows)}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _fv(val, unit="", decimals=2):
    """Format a numeric value; returns None if unavailable."""
    if val is None:
        return None
    try:
        f = float(val)
        import math
        if math.isnan(f):
            return None
        return f"{f:.{decimals}f}{unit}"
    except (TypeError, ValueError):
        s = str(val).strip()
        return s if s else None


def _render_idx(val, unit="", decimals=2):
    v = _fv(val, unit, decimals)
    if v is None:
        return '<span class="idx-na">tidak tersedia</span>'
    return f'<span class="idx-val">{v}</span>'


_RISK_CSS = {
    "Rendah": "rc-rendah",
    "Sedang": "rc-sedang",
    "Tinggi": "rc-tinggi",
    "Sangat Tinggi": "rc-sangat-tinggi",
}

# ─────────────────────────────────────────────────────────────────────────────
# ── NAVBAR ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<nav class="navbar">
    <div class="navbar-inner">
        <span class="nav-brand">Flood Risk Assessment</span>
        <div class="nav-links">
            <a href="#beranda">Beranda</a>
            <a href="#prediksi">Prediksi</a>
            <a href="#riwayat">Riwayat</a>
            <a href="#tentang">Tentang</a>
        </div>
    </div>
</nav>
<div class="nav-spacer"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION: BERANDA ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a id="beranda" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section">
<div class="page-section-inner">
<div class="hero-wrap">
    <div class="hero-eyebrow">Sistem Prediksi Risiko Banjir Kota Padang</div>
    <h1 class="hero-title">Sistem Penilaian<br><em>Risiko Banjir</em></h1>
    <p class="hero-body">
        Prakiraan tingkat risiko banjir di Kota Padang untuk tanggal pilihan Anda, berdasarkan data cuaca dan kondisi atmosfer selama 30 hari terakhir.
    </p>
    <a href="#prediksi" class="hero-cta">Mulai Prediksi &nbsp;→</a>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-val">LSTM</div>
            <div class="hero-stat-lbl">Metode Prediksi</div>
        </div>
        <div class="hero-stat" style="padding-left:32px;">
            <div class="hero-stat-val">30 Hari</div>
            <div class="hero-stat-lbl">Rentang Data yang Digunakan</div>
        </div>
        <div class="hero-stat" style="padding-left:32px;">
            <div class="hero-stat-val">8 Variabel</div>
            <div class="hero-stat-lbl">Variabel yang Dianalisis</div>
        </div>
        <div class="hero-stat" style="padding-left:32px;">
            <div class="hero-stat-val">4 Tingkat</div>
            <div class="hero-stat-lbl">Tingkat Risiko</div>
        </div>
        <div class="hero-stat" style="padding-left:32px;">
            <div class="hero-stat-val">Ogimet · Wyoming</div>
            <div class="hero-stat-lbl">Sumber Data</div>
        </div>
    </div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION: PREDIKSI ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a id="prediksi" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section">
<div class="page-section-inner">
    <div class="s-eyebrow">Prediksi</div>
    <div class="s-title">Prediksi Risiko Banjir</div>
    <div class="s-body">
       Pilih tanggal yang ingin diprediksi. Sistem akan otomatis mengambil data cuaca dan atmosfer 30 hari sebelumnya. Anda tidak perlu mengisi data apa pun secara manual.
    </div>
""", unsafe_allow_html=True)

# Form
col_input, col_pad = st.columns([2, 3])
with col_input:
    target_date_widget = st.date_input(
        "Tanggal Prediksi",
        value=None,
        key="date_picker",
        help="Sistem akan menggunakan data 30 hari sebelum tanggal ini untuk membuat prediksi.",
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    submit_disabled = (target_date_widget is None) or st.session_state.running
    submit_clicked = st.button(
        "Prediksi Sekarang",
        disabled=submit_disabled,
        key="btn_submit",
    )

# Trigger: simpan tanggal dan rerun untuk masuk ke progress loop
if submit_clicked and target_date_widget is not None:
    st.session_state.target_date = target_date_widget.strftime("%Y-%m-%d")
    st.session_state.running = True
    st.session_state.result = None
    st.session_state.csv_path = None
    st.rerun()

# ── LIVE PROGRESS ──
if st.session_state.running and st.session_state.target_date:
    tgt = st.session_state.target_date
    st.markdown(
        f'<div style="margin-top:32px; font-size:0.7rem; font-weight:700; '
        f'letter-spacing:0.18em; text-transform:uppercase; color:#aaa;">'
        f'Memproses Tanggal {tgt}</div>',
        unsafe_allow_html=True,
    )
    tracker_ph = st.empty()

    # Init statuses: stage 1 immediately 'running'
    statuses = [
        {"status": "running" if i == 0 else "pending", "elapsed": None}
        for i in range(len(PIPELINE_STAGES))
    ]
    tracker_ph.markdown(_render_tracker(statuses), unsafe_allow_html=True)

    # Setup
    log_q: queue.Queue = queue.Queue()
    result_ref: dict = {"result": None}
    orig_stdout = sys.stdout

    # Create thread (without starting yet so we can get ident after start)
    thread = threading.Thread(
        target=_inference_worker,
        args=(tgt, result_ref, log_q),
        daemon=True,
    )

    # Install thread-aware writer BEFORE starting thread
    # (Python GIL ensures this assignment is seen by the new thread)
    thread.start()
    sys.stdout = ThreadAwareWriter(orig_stdout, log_q, thread.ident)

    # ── UPDATE LOOP ──
    done = False
    while not done:
        try:
            while True:
                line = log_q.get_nowait()
                if line is None:
                    done = True
                    break
                _parse_stage(line, statuses)
                tracker_ph.markdown(_render_tracker(statuses), unsafe_allow_html=True)
        except queue.Empty:
            pass

        if not done:
            tracker_ph.markdown(_render_tracker(statuses), unsafe_allow_html=True)
            time.sleep(0.2)

    # Restore stdout
    sys.stdout = orig_stdout
    thread.join(timeout=10)

    # Finalize: mark any lingering 'running' as done
    for s in statuses:
        if s["status"] == "running":
            s["status"] = "done"
    tracker_ph.markdown(_render_tracker(statuses), unsafe_allow_html=True)

    # Persist result to session state
    final_result = result_ref["result"]
    st.session_state.result = final_result
    st.session_state.running = False

    csv_candidate = f"inference_features_{tgt}.csv"
    if os.path.exists(csv_candidate):
        st.session_state.csv_path = csv_candidate

    # Add to history (avoid duplicates)
    if final_result and final_result.get("status") == "SUCCESS":
        probs = final_result.get("probabilities", {})
        entry = {
            "Tanggal": tgt,
            "Prediksi": final_result.get("predicted_class_name", "-"),
            "Rendah %": f"{probs.get('Rendah', 0)*100:.1f}",
            "Sedang %": f"{probs.get('Sedang', 0)*100:.1f}",
            "Tinggi %": f"{probs.get('Tinggi', 0)*100:.1f}",
            "Sangat Tinggi %": f"{probs.get('Sangat Tinggi', 0)*100:.1f}",
        }
        if not any(h["Tanggal"] == tgt for h in st.session_state.history):
            st.session_state.history.insert(0, entry)

# ── RESULTS ──
if st.session_state.result and not st.session_state.running:
    res = st.session_state.result
    tgt_date = res.get("target_date", "")

    if res.get("status") == "SUCCESS":
        pred = res.get("predicted_class_name", "")
        rc = _RISK_CSS.get(pred, "rc-rendah")
        probs = res.get("probabilities", {})

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-eyebrow">Hasil Prediksi untuk {tgt_date}</div>
            <div class="result-class {rc}">{pred}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability cards
        cards_html = ""
        for cls in CLASS_NAMES:
            p = probs.get(cls, 0.0)
            is_active = "active" if cls == pred else ""
            cards_html += f"""
            <div class="prob-card {is_active}">
                <div class="prob-card-lbl">{cls}</div>
                <div class="prob-card-val">{p*100:.1f}%</div>
                <div class="pbar-bg"><div class="pbar-fill" style="width:{p*100:.1f}%"></div></div>
            </div>"""
        st.markdown(f'<div class="prob-grid">{cards_html}</div>', unsafe_allow_html=True)

        # Atmospheric indices from generated CSV
        csv_path = st.session_state.csv_path
        df_feat = None
        if csv_path and os.path.exists(csv_path):
            try:
                df_feat = pd.read_csv(csv_path)
            except Exception:
                pass

        if df_feat is not None and not df_feat.empty:
            last = df_feat.iloc[-1]
            d1 = str(last.get("date", "D-1"))

            st.markdown(
                f'<div class="indices-label">Kondisi Atmosfer Terakhir yang Digunakan ({d1})</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div class="indices-grid">
                <div class="idx-card">
                    <div class="idx-lbl">Curah Hujan</div>
                    {_render_idx(last.get('rr'), ' mm')}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">Suhu Udara Rata-rata</div>
                    {_render_idx(last.get('tavg'), ' °C')}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">Kelembapan Udara</div>
                    {_render_idx(last.get('rh'), ' %')}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">CIN (Convective Inhibition)</div>
                    {_render_idx(last.get('cin'), ' J/kg')}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">K-Index</div>
                    {_render_idx(last.get('kindex'))}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">Lifted Index (LI)</div>
                    {_render_idx(last.get('li'))}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">Total Totals (TT)</div>
                    {_render_idx(last.get('tt'))}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">SWEAT Index</div>
                    {_render_idx(last.get('sweat'))}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Lihat Data 30 Hari Terakhir yang Digunakan)"):
                st.dataframe(df_feat, use_container_width=True, hide_index=True)

            col_dl, _ = st.columns([1, 3])
            with col_dl:
                with open(csv_path, "rb") as fh:
                    st.download_button(
                        "Unduh CSV",
                        data=fh,
                        file_name=f"features_{tgt_date}.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Detail parameter atmosfer tidak tersedia untuk tanggal ini).")

    else:
        reasons = res.get("reasons", ["Alasan tidak diketahui."])
        st.error("Prediksi tidak dapat dilakukan untuk tanggal ini:")
        for r in reasons:
            st.markdown(f"<small>• {r}</small>", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("Coba Tanggal Lain", key="btn_reset"):
            st.session_state.result = None
            st.session_state.target_date = None
            st.session_state.csv_path = None
            st.rerun()

# Close prediksi section div
st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION: RIWAYAT ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a id="riwayat" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section">
<div class="page-section-inner">
    <div class="s-eyebrow">Riwayat</div>
    <div class="s-title">Riwayat Prediksi</div>
    <div class="s-body">Daftar prediksi yang sudah Anda lihat selama kunjungan ini. Riwayat akan hilang jika halaman ditutup atau dimuat ulang.</div>
""", unsafe_allow_html=True)

hist = st.session_state.history
if hist:
    st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div class="empty-state">Belum ada prediksi yang dilakukan. Pilih tanggal di atas untuk memulai.</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION: TENTANG ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a id="tentang" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section">
<div class="page-section-inner">
    <div class="s-eyebrow">Tentang</div>
    <div class="s-title">Metodologi &amp; Sistem</div>
    <div class="s-body">
       Sistem ini bekerja melalui tiga tahap utama: pengumpulan dan penggabungan data, pengolahan data hingga menghasilkan prediksi, serta penyajian hasil kepada pengguna melalui halaman yang sedang Anda gunakan ini.
    </div>
    <div class="about-grid">
        <div class="about-block">
            <div class="about-block-title">Tahapan Proses Prediksi</div>
            <ul class="about-list">
                <li>Mengambil data cuaca harian dari Ogimet (curah hujan, suhu udara, kelembapan)</li>
                <li>Mengambil data atmosfer atas dari Wyoming Upper Air (pengukuran pukul 12.00 dan 00.00 UTC)</li>
                <li>Menghitung indeks atmosfer menggunakan SounderPy dan SHARPpy</li>
                <li>Menggabungkan seluruh data ke dalam satu susunan harian yang berurutan</li>
                <li>Membersihkan dan melengkapi data yang kosong menggunakan nilai tengah (median) bulanan</li>
                <li>Menyusun data 30 hari terakhir dan memeriksa kelayakannya melalui 9 tahap pemeriksaan kualitas data</li>
                <li>Menjalankan model LSTM untuk menentukan kategori risiko banjir yang paling mungkin terjadi</li>
            </ul>
        </div>
        <div class="about-block">
            <div class="about-block-title">Variabel yang Digunakan Model (8 Variabel)</div>
            <ul class="about-list">
                <li>Curah Hujan Harian (mm)</li>
                <li>Suhu Udara Rata-rata (°C)</li>
                <li>Kelembapan Relatif (%)</li>
                <li>CIN — Convective Inhibition (J/kg)</li>
                <li>K-Index</li>
                <li>Lifted Index (LI)</li>
                <li>Total Totals Index (TT)</li>
                <li>SWEAT Index</li>
            </ul>
        </div>
        <div class="about-block">
            <div class="about-block-title">Sumber Data</div>
            <ul class="about-list">
                <li>Ogimet — data permukaan harian BMKG Stasiun 96163</li>
                <li>Wyoming Upper Air — data sounding atmosfer pukul 12.00 dan 00.00 UTC</li>
                <li>SounderPy dan SHARPpy — digunakan untuk mengolah data sounding menjadi indeks atmosfer</li>
                <li>Data historis tahun 2018–2024</li>
            </ul>
        </div>
        <div class="about-block">
            <div class="about-block-title">Spesifikasi Model</div>
            <ul class="about-list">
                <li>Arsitektur: LSTM berlapis (multi-layer), dengan 4 kategori keluaran</li>
                <li>Kategori hasil prediksi: Rendah · Sedang · Tinggi · Sangat Tinggi</li>
                <li>Rentang data yang digunakan: 30 hari sebelum tanggal prediksi</li>
                <li>Normalisasi data: menggunakan StandardScaler, dengan parameter yang dihitung sekali saat pelatihan model dan diterapkan secara konsisten pada setiap prediksi baru.</li>
                <li>Format penyimpanan model: Keras (.keras)</li>
            </ul>
        </div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div style="border-top:1px solid #e4e4e0; padding:32px 64px; display:flex; justify-content:space-between; align-items:center;">
    <span style="font-size:0.7rem; color:#bbb; letter-spacing:0.1em; text-transform:uppercase;">
        Sistem Penilaian Risiko Banjir Kota Padang
    </span>
    <span style="font-size:0.7rem; color:#bbb; letter-spacing:0.06em;">
        © 2026 Imam. All rights reserved.
    </span>
</div>
""", unsafe_allow_html=True)
