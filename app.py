import streamlit as st
import pickle
import json
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import io
import csv
import time
import html
from pathlib import Path

import pandas as pd

from clinic_tools import (
    TIERS, add_scan, build_report_pdf, delete_patient, discover_models, get_history,
    gradcam_overlay, list_patients, load_model_files, priority_sort, triage_tier,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --cyan:   #00F5FF;
    --green:  #00FF88;
    --red:    #FF3860;
    --orange: #FF8C00;
    --yellow: #FFD700;
    --bg:     #020B18;
    --card:   #0A1F35;
    --border: #0D3B66;
    --text:   #CDD6F4;
    --dim:    #6C7086;
}

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* ── Main title ── */
.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--cyan);
    text-align: center;
    letter-spacing: 6px;
    text-shadow: 0 0 30px rgba(0,245,255,0.5);
    margin-bottom: 0.2rem;
}
.main-sub {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.85rem;
    color: var(--dim);
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 1.5rem;
}

/* ── Cards ── */
.scan-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--cyan);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 0.8rem 0;
}

/* ── Result banners ── */
.result-normal {
    background: linear-gradient(135deg, #003322, #004433);
    border: 2px solid var(--green);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,255,136,0.3);
}
.result-pneumonia {
    background: linear-gradient(135deg, #2D0014, #3D0020);
    border: 2px solid var(--red);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(255,56,96,0.4);
    animation: pulse-red 2s infinite;
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 20px rgba(255,56,96,0.4); }
    50%  { box-shadow: 0 0 40px rgba(255,56,96,0.7); }
    100% { box-shadow: 0 0 20px rgba(255,56,96,0.4); }
}
.result-label {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 4px;
}

/* ── Metric badges ── */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-badge {
    flex: 1;
    background: rgba(0,245,255,0.07);
    border: 1px solid rgba(0,245,255,0.25);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    text-align: center;
    min-width: 100px;
}
.metric-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.55rem;
    color: var(--dim);
    letter-spacing: 2px;
    display: block;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--cyan);
}

/* ── Probability bars ── */
.prob-container { margin: 0.5rem 0; }
.prob-label {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.8rem;
    color: var(--dim);
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
.prob-track {
    width: 100%;
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    height: 14px;
    overflow: hidden;
}
.prob-fill-g {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00FF88, #00CC66);
    transition: width 0.8s ease;
}
.prob-fill-r {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #FF3860, #CC0033);
    transition: width 0.8s ease;
}
.prob-pct {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    text-align: right;
    margin-top: 2px;
}

/* ── Severity meter ── */
.sev-section { margin: 1rem 0; }
.sev-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    color: var(--dim);
    letter-spacing: 3px;
    margin-bottom: 0.5rem;
}
.sev-track {
    width: 100%;
    height: 18px;
    border-radius: 9px;
    background: linear-gradient(90deg, #00FF88 0%, #FFD700 35%, #FF8C00 65%, #FF3860 100%);
    position: relative;
    box-shadow: 0 0 10px rgba(0,245,255,0.2);
}
.sev-needle {
    position: absolute;
    top: -5px;
    width: 4px;
    height: 28px;
    background: white;
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(255,255,255,0.8);
    transform: translateX(-50%);
}
.sev-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 0.5rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #050F1C !important;
    border-right: 1px solid var(--border);
}
.sidebar-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    color: var(--cyan);
    letter-spacing: 3px;
    margin-bottom: 0.8rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-ok  { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-err { background: var(--red);   box-shadow: 0 0 6px var(--red);   }
.meta-row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid rgba(13,59,102,0.5);
    font-size: 0.82rem;
}
.meta-key { color: var(--dim); font-family: 'Exo 2', sans-serif; }
.meta-val { color: var(--cyan); font-family: 'Orbitron', monospace; font-size: 0.75rem; }

/* ── Batch cards ── */
.batch-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem;
    margin-bottom: 0.5rem;
    text-align: center;
}
.batch-normal   { border-top: 3px solid var(--green); }
.batch-pneumonia{ border-top: 3px solid var(--red); }

/* ── Summary badges ── */
.summary-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
.summary-badge {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.8rem;
    text-align: center;
    min-width: 90px;
}
.summary-num {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 900;
}
.summary-lbl {
    font-size: 0.7rem;
    color: var(--dim);
    letter-spacing: 2px;
    font-family: 'Orbitron', monospace;
}

/* ── Streamlit tweaks ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--dim) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,245,255,0.12) !important;
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,245,255,0.03) !important;
    border: 1.5px dashed var(--cyan) !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0D3B66, #1565C0) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    border-radius: 6px !important;
}
.stSelectbox label, .stFileUploader label {
    color: var(--dim) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
}
.stExpander {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def build_transform(input_size, mean, std):
    """Build preprocessing pipeline matching training conditions."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def predict(model, image, transform, class_names):
    """Run inference on a single PIL Image. Returns label, confidence, all probs."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    pred_idx   = probs.argmax().item()
    pred_label = class_names[pred_idx]
    confidence = probs[pred_idx].item() * 100
    all_probs  = {cls: probs[i].item() * 100 for i, cls in enumerate(class_names)}
    return pred_label, confidence, all_probs


def get_severity(pneu_prob):
    """Map pneumonia probability to clinical severity level."""
    if pneu_prob < 30:   return "LOW",      "#00FF88"
    elif pneu_prob < 60: return "MODERATE",  "#FFD700"
    elif pneu_prob < 80: return "HIGH",      "#FF8C00"
    else:                return "CRITICAL",  "#FF3860"


def apply_processing(img, mode):
    """Apply selected image enhancement. Only affects display, not model input."""
    if mode == "Original":       return img
    if mode == "High Contrast":  return ImageEnhance.Contrast(img).enhance(2.2)
    if mode == "Sharpen":        return img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    if mode == "Edge Highlight": return img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    if mode == "Equalized":      return ImageOps.equalize(img.convert("RGB"))
    return img


def prob_bar_html(label, pct, fill_class):
    return f"""
    <div class="prob-container">
        <div class="prob-label">{label}</div>
        <div class="prob-track">
            <div class="{fill_class}" style="width:{pct:.1f}%"></div>
        </div>
        <div class="prob-pct" style="color:{'#00FF88' if fill_class=='prob-fill-g' else '#FF3860'}">
            {pct:.1f}%
        </div>
    </div>"""


def severity_meter_html(pneu_prob, sev_label, sev_color):
    needle_pct = min(pneu_prob, 97)
    return f"""
    <div class="sev-section">
        <div class="sev-title">RISK ASSESSMENT METER</div>
        <div class="sev-track">
            <div class="sev-needle" style="left:{needle_pct}%"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.65rem;
                    color:var(--dim); font-family:'Orbitron',monospace; margin-top:4px;">
            <span>LOW</span><span>MODERATE</span><span>HIGH</span><span>CRITICAL</span>
        </div>
        <div>
            <span class="sev-badge"
                  style="background:rgba(0,0,0,0.4); border:2px solid {sev_color};
                         color:{sev_color}; margin-top:0.5rem;">
                SEVERITY: {sev_label}
            </span>
        </div>
    </div>"""


# ── Extra styles (hackathon additions) ────────────────────────────────────────
st.markdown("""
<style>
.result-uncertain {
    background: linear-gradient(135deg, #2B2300, #3A3000);
    border: 2px solid var(--yellow);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(255,215,0,0.35);
}
.tier-chip {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 20px;
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    border: 2px solid;
    background: rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)



# ── Themes / contrast ─────────────────────────────────────────────────────────
THEMES = {
    "Neon Dark (default)": dict(
        field_bg="#0A1F35", field_text="#CDD6F4",
        light=False, bg="#020B18", card="#0A1F35", border="#0D3B66", text="#CDD6F4", dim="#6C7086",
        cyan="#00F5FF", green="#00FF88", red="#FF3860", orange="#FF8C00", yellow="#FFD700",
        sidebar="#050F1C", track="rgba(255,255,255,0.07)",
        ok_bg="linear-gradient(135deg,#003322,#004433)", bad_bg="linear-gradient(135deg,#2D0014,#3D0020)",
        warn_bg="linear-gradient(135deg,#2B2300,#3A3000)",
        btn_bg="linear-gradient(135deg,#0D3B66,#1565C0)", btn_text="#00F5FF",
        sel_bg="rgba(0,245,255,0.12)", bw="1px"),
    "Soft Dark (low contrast)": dict(
        field_bg="#243147", field_text="#E6ECF8",
        light=False, bg="#1B2433", card="#243147", border="#3A4B66", text="#D8DFEE", dim="#9AA7BF",
        cyan="#7FD8E8", green="#86E0B0", red="#F594A4", orange="#F2B36B", yellow="#EBD58A",
        sidebar="#151D2B", track="rgba(255,255,255,0.10)",
        ok_bg="#1F3D33", bad_bg="#46242E", warn_bg="#43391D",
        btn_bg="#33507A", btn_text="#FFFFFF", sel_bg="rgba(127,216,232,0.15)", bw="1px"),
    "High Contrast Dark": dict(
        field_bg="#000000", field_text="#FFFFFF",
        light=False, bg="#000000", card="#000000", border="#FFFFFF", text="#FFFFFF", dim="#E6E6E6",
        cyan="#00FFFF", green="#00FF66", red="#FF6666", orange="#FFB000", yellow="#FFFF00",
        sidebar="#000000", track="rgba(255,255,255,0.25)",
        ok_bg="#003A1A", bad_bg="#4A0000", warn_bg="#3F3F00",
        btn_bg="#000000", btn_text="#FFFF00", sel_bg="rgba(255,255,0,0.25)", bw="2px"),
    "Light": dict(
        field_bg="#14233A", field_text="#FFFFFF",
        light=True, bg="#F4F7FB", card="#FFFFFF", border="#B7C6DA", text="#14233A", dim="#4B5B75",
        cyan="#0B5CAD", green="#0A7A45", red="#C0143C", orange="#B35F00", yellow="#8A6D00",
        sidebar="#E8EEF7", track="rgba(0,0,0,0.12)",
        ok_bg="#E1F5EA", bad_bg="#FCE4EA", warn_bg="#FFF3C4",
        btn_bg="#0B5CAD", btn_text="#FFFFFF", sel_bg="rgba(11,92,173,0.14)", bw="1px"),
    "High Contrast Light": dict(
        field_bg="#000000", field_text="#FFFFFF",
        light=True, bg="#FFFFFF", card="#FFFFFF", border="#000000", text="#000000", dim="#1F1F1F",
        cyan="#0000CC", green="#006400", red="#B00000", orange="#8A4500", yellow="#5C4600",
        sidebar="#FFFFFF", track="rgba(0,0,0,0.25)",
        ok_bg="#D8F5DF", bad_bg="#FFD9D9", warn_bg="#FFF0A8",
        btn_bg="#000000", btn_text="#FFFFFF", sel_bg="rgba(0,0,204,0.18)", bw="2px"),
}
DEFAULT_THEME = "Neon Dark (default)"

# neon colours hard-coded in the HTML -> theme variable they should follow
_NEON = {"#00F5FF": "cyan", "#00FF88": "green", "#66FFB2": "green", "#FF3860": "red",
         "#FF6B8A": "red", "#FFD700": "yellow", "#FFE680": "yellow", "#FF8C00": "orange"}


def build_theme_css(name):
    t = THEMES[name]
    css = ":root{" + ";".join(
        f"--{k.replace('_', '-')}:{v}" for k, v in t.items() if k != "light") + "}\n"
    # tables are used in every theme
    css += """
    .ps-table{width:100%; border-collapse:collapse; font-size:0.8rem; color:var(--text);}
    .ps-table th{background:var(--card); color:var(--cyan); text-align:left; padding:6px 10px;
        border-bottom:2px solid var(--border); font-family:'Orbitron',monospace; font-size:0.6rem;
        letter-spacing:2px; white-space:nowrap;}
    .ps-table td{padding:6px 10px; border-bottom:1px solid var(--border); white-space:nowrap;}
    """
    if name == DEFAULT_THEME:
        return "<style>" + css + "</style>"      # leave the original look untouched

    css += """
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background: var(--bg) !important;}
    section[data-testid="stSidebar"] {background: var(--sidebar) !important; border-right:1px solid var(--border) !important;}
    .stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] {color: var(--text);}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color: var(--dim) !important;}
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {color: var(--dim) !important;}
    div[data-baseweb="select"], div[data-baseweb="select"] div,
    div[data-baseweb="input"], div[data-baseweb="input"] > div, div[data-baseweb="base-input"],
    [data-testid="stTextInputRootElement"], [data-testid="stTextInputRootElement"] > div {
        background: var(--field-bg) !important; background-color: var(--field-bg) !important;
        border-color: var(--border) !important;}
    div[data-baseweb="select"] *, div[data-baseweb="input"] input, [data-testid="stTextInputRootElement"] input,
    textarea {color: var(--field-text) !important; -webkit-text-fill-color: var(--field-text) !important;
        caret-color: var(--field-text) !important;}
    input::placeholder, textarea::placeholder {color: var(--field-text) !important; opacity: 0.6 !important;}
    div[data-baseweb="select"] svg {fill: var(--field-text) !important;}
    div[data-baseweb="popover"], div[data-baseweb="popover"] div, [data-baseweb="menu"], [data-baseweb="menu"] li,
    ul[role="listbox"], li[role="option"] {
        background: var(--field-bg) !important; background-color: var(--field-bg) !important;
        color: var(--field-text) !important;}
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {background: var(--sel-bg) !important;}
    .stButton > button, .stDownloadButton > button, [data-testid="stDownloadButton"] button,
    div[data-testid="stFileUploaderDropzone"] button {
        background: var(--btn-bg) !important; color: var(--btn-text) !important;
        border: 1px solid var(--cyan) !important;}
    .stButton > button p, .stDownloadButton > button p, [data-testid="stDownloadButton"] button p {color: var(--btn-text) !important;}
    div[data-testid="stFileUploaderDropzone"] {background: var(--card) !important; border:1.5px dashed var(--cyan) !important;}
    div[data-testid="stFileUploaderDropzone"] small, div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] div {color: var(--text) !important;}
    .stExpander, [data-testid="stExpander"] details {background: var(--card) !important; border:1px solid var(--border) !important;}
    .stExpander summary, .stExpander summary *, [data-testid="stExpander"] summary * {color: var(--text) !important;}
    .stTabs [aria-selected="true"] {background: var(--sel-bg) !important;}
    .stTabs [data-baseweb="tab"] {color: var(--dim) !important;}
    .stTabs [aria-selected="true"] {color: var(--cyan) !important;}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] details > summary {background: var(--card) !important;}
    label[data-baseweb="radio"]:has(input[aria-checked="false"]) > div:first-child {
        background: var(--card) !important; border-color: var(--border) !important;}
    .result-normal {background: var(--ok-bg) !important;}
    .result-pneumonia {background: var(--bad-bg) !important;}
    .result-uncertain {background: var(--warn-bg) !important;}
    .prob-track {background: var(--track) !important;}
    .sev-badge, .tier-chip {background: var(--card) !important;}
    .sev-needle {background: var(--text) !important; box-shadow: 0 0 6px rgba(0,0,0,0.5) !important;}
    .scan-card, .summary-badge, .batch-card, .metric-badge {border-width: var(--bw) !important;}
    """
    if t["light"]:
        css += ".main-title {text-shadow:none !important;} .metric-badge{background: var(--card) !important;}\n"
    # make every hard-coded neon colour follow the chosen theme
    for hexc, var in _NEON.items():
        css += f'[style*="color:{hexc}" i] {{color: var(--{var}) !important;}}\n'
        css += f'.tier-chip[style*="{hexc}" i], .sev-badge[style*="{hexc}" i] {{border-color: var(--{var}) !important;}}\n'
        css += f'.batch-card[style*="{hexc}" i] {{border-top-color: var(--{var}) !important;}}\n'
    return "<style>" + css + "</style>"


def html_table(df):
    """Theme-aware HTML table (st.dataframe ignores our theme)."""
    tier_colors = {v[0]: v[1] for v in TIERS.values()}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = ""
        for c in df.columns:
            val = html.escape(str(row[c]))
            if c == "triage" and row[c] in tier_colors:
                val = f'<b style="color:{tier_colors[row[c]]};">{val}</b>'
            cells += f"<td>{val}</td>"
        body += f"<tr>{cells}</tr>"
    return (f'<div style="overflow-x:auto;"><table class="ps-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


# ── Plug-in model registry ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing neural network…")
def load_model_from(model_path, meta_path):
    return load_model_files(model_path, meta_path)


registry = discover_models()

with st.sidebar:
    st.markdown('<div class="sidebar-title">🎨 DISPLAY</div>', unsafe_allow_html=True)
    theme_name = st.selectbox("Theme / contrast", list(THEMES.keys()), key="theme_pick")
    view_mode = st.selectbox("Scan view", ["Tabs", "Side by side", "Stacked"], key="view_pick",
                             help="Tabs: scan and heatmap in tabs. Side by side: both images next to each "
                                  "other. Stacked: one under the other.")
    st.markdown('<div class="sidebar-title" style="margin-top:1rem;">🧩 ACTIVE MODEL</div>',
                unsafe_allow_html=True)
    if registry:
        active_name = st.selectbox("Active model", list(registry.keys()), label_visibility="collapsed")
    else:
        active_name = None
        st.warning("No model found. Add `pneumonia_model.pkl`, or a folder inside `models/`.")

st.markdown(build_theme_css(theme_name), unsafe_allow_html=True)

if active_name:
    model, metadata = load_model_from(*registry[active_name])
else:
    model, metadata = None, {}

input_size  = metadata.get("input_size", 224)
mean        = metadata.get("imagenet_mean", [0.485, 0.456, 0.406])
std_norm    = metadata.get("imagenet_std",  [0.229, 0.224, 0.225])
class_names = metadata.get("class_names",   ["NORMAL", "PNEUMONIA"])
POS         = metadata.get("positive_class", "PNEUMONIA")      # the "disease" class
if POS not in class_names:
    POS = class_names[-1]
DISEASE     = metadata.get("disease_name", POS.title())
CAM_LAYER   = metadata.get("gradcam_layer", "layer4")
transform   = build_transform(input_size, mean, std_norm)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title" style="margin-top:1rem;">⚡ SYSTEM STATUS</div>',
                unsafe_allow_html=True)
    model_ok = model is not None
    meta_ok  = bool(metadata)
    st.markdown(f"""
    <div style="font-family:'Exo 2',sans-serif; font-size:0.85rem; margin-bottom:1rem;">
        <div style="margin-bottom:6px;">
            <span class="status-dot {'status-ok' if model_ok else 'status-err'}"></span>
            <span style="color:{'#00FF88' if model_ok else '#FF3860'}">
                {'MODEL LOADED' if model_ok else 'MODEL NOT FOUND'}
            </span>
        </div>
        <div>
            <span class="status-dot {'status-ok' if meta_ok else 'status-err'}"></span>
            <span style="color:{'#00FF88' if meta_ok else '#FF3860'}">
                {'METADATA OK' if meta_ok else 'METADATA MISSING'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if metadata:
        st.markdown('<div class="sidebar-title" style="margin-top:1rem;">📊 MODEL METRICS</div>',
                    unsafe_allow_html=True)
        metrics = [
            ("ARCHITECTURE",  metadata.get("architecture", "ResNet18")),
            ("DETECTS",       DISEASE.upper()),
            ("TEST ACCURACY", f"{metadata.get('test_accuracy', '-')}%"),
            ("AUC-ROC",       str(metadata.get("test_auc_roc", "-"))),
            ("EPOCHS",        str(metadata.get("epochs_trained", "-"))),
            ("INPUT SIZE",    f"{input_size}×{input_size}"),
        ]
        rows_html = "".join(
            f'<div class="meta-row"><span class="meta-key">{k}</span><span class="meta-val">{v}</span></div>'
            for k, v in metrics
        )
        st.markdown(f'<div style="margin-bottom:1rem;">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🚦 TRIAGE & HISTORY</div>', unsafe_allow_html=True)
    conf_thresh = st.slider(
        "Min. confidence (%) to be 'sure'", 50, 95, 80, 5,
        help="Scans below this confidence are flagged REVIEW (show to a doctor). "
             "This is a simple confidence cut-off, not a calibrated uncertainty estimate.",
    )
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    with st.expander("🔥 Heatmap settings"):
        _target = st.radio("Explain which class?", ["Predicted class", f"{DISEASE} (disease)"],
                           help="'Predicted class' shows evidence for whatever the model decided. "
                                f"'{DISEASE} (disease)' always shows where it sees signs of {DISEASE.lower()}.")
        cam_target = "disease" if _target.startswith(DISEASE) else "pred"
        cam_thr = st.slider("Hide weak areas (%)", 0, 80, 35, 5,
                            help="Higher = only the strongest regions are coloured (cleaner map).")
        cam_alpha = st.slider("Heat intensity", 0.2, 1.0, 0.65, 0.05)
        finer_cam = st.checkbox("Finer detail (earlier layer)", value=False,
                                help="Uses layer3 (14×14) instead of layer4 (7×7). Sharper but noisier.")
    patient_id = st.text_input(
        "Patient ID (saves scans to history)", placeholder="e.g. P-001",
        help="Use an ID or code, not a real name. Data stays in a local SQLite file.",
    ).strip()

    st.markdown('<div class="sidebar-title" style="margin-top:1rem;">🔧 IMAGE PROCESSING</div>',
                unsafe_allow_html=True)
    processing_mode = st.selectbox(
        "Enhancement Mode",
        ["Original", "High Contrast", "Sharpen", "Edge Highlight", "Equalized"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:var(--dim); text-align:center; line-height:1.8;">
        <span style="color:#FF3860;">⚠</span> FOR RESEARCH USE ONLY<br>
        Not a certified medical device.<br>
        Always consult a physician.
    </div>
    """, unsafe_allow_html=True)


# ── Analysis pipeline (prediction + severity + uncertainty + Grad-CAM) ────────
def make_cam(rgb, label):
    idx = class_names.index(POS) if cam_target == "disease" else class_names.index(label)
    layers = [CAM_LAYER]
    if finer_cam and CAM_LAYER == "layer4":
        layers.insert(0, "layer3")
    for layer in layers:
        try:
            return gradcam_overlay(model, rgb, transform, idx, layer,
                                   alpha=cam_alpha, threshold=cam_thr / 100)
        except Exception:
            continue          # e.g. a plug-in model whose layer name differs
    return None


def analyze(img, with_cam=True):
    rgb = img.convert("RGB")
    label, conf, probs = predict(model, rgb, transform, class_names)
    pos_prob = probs.get(POS, 0.0)
    sev_label, sev_color = get_severity(pos_prob)
    is_pos    = label == POS
    uncertain = conf < conf_thresh
    tier      = triage_tier(is_pos, uncertain)
    tier_label, tier_color = TIERS[tier]

    cam = make_cam(rgb, label) if with_cam else None

    return {
        "label": label, "conf": conf, "probs": probs, "pos_prob": pos_prob,
        "sev_label": sev_label, "sev_color": sev_color,
        "is_pos": is_pos, "uncertain": uncertain,
        "tier": tier, "tier_label": tier_label, "tier_color": tier_color,
        "cam": cam,
        "cam_class": POS if cam_target == "disease" else label,
    }


def banner_html(res):
    if res["uncertain"]:
        return f"""
        <div class="result-uncertain">
            <div class="result-label" style="color:#FFD700;">⚠ NOT SURE</div>
            <div style="font-size:0.8rem; color:#FFE680; margin-top:0.4rem; font-family:'Exo 2',sans-serif;">
                Confidence {res['conf']:.1f}% is below {conf_thresh}% — please show this scan to a doctor
            </div>
        </div>"""
    if res["is_pos"]:
        return f"""
        <div class="result-pneumonia">
            <div class="result-label" style="color:#FF3860;">⚠ {DISEASE.upper()} DETECTED</div>
            <div style="font-size:0.8rem; color:#FF6B8A; margin-top:0.4rem; font-family:'Exo 2',sans-serif;">
                Consult a radiologist immediately
            </div>
        </div>"""
    return f"""
        <div class="result-normal">
            <div class="result-label" style="color:#00FF88;">✓ NORMAL</div>
            <div style="font-size:0.8rem; color:#66FFB2; margin-top:0.4rem; font-family:'Exo 2',sans-serif;">
                No {DISEASE.lower()} indicators detected
            </div>
        </div>"""


def notes_html(res):
    c, sev = res["conf"], res["sev_label"]
    disclaimer = """<b style="color:#FFD700;">⚠ Disclaimer:</b> This AI tool is for research
        purposes only and is NOT a substitute for professional medical diagnosis."""
    if res["uncertain"]:
        body = f"""<b style="color:#FFD700;">⚠ Model is not sure</b><br><br>
        Confidence is only <b>{c:.1f}%</b> (threshold {conf_thresh}%).<br>
        • Do not rely on this result — refer the scan to a doctor or radiologist.<br>
        • Consider a repeat or better-quality image."""
    elif res["is_pos"]:
        extra = ("• Laboratory investigations: CBC, CRP, sputum culture.<br>"
                 if DISEASE.lower() == "pneumonia" else "")
        body = f"""<b style="color:#FF3860;">⚠ {DISEASE} Indicators Present</b><br><br>
        Confidence: <b>{c:.1f}%</b>, Severity: <b style="color:{res['sev_color']};">{sev}</b>.<br><br>
        <b style="color:#00F5FF;">Recommended Actions:</b><br>
        • Urgent consultation with a qualified radiologist or physician.<br>
        • Clinical correlation with the patient's symptoms.<br>
        • Consider further imaging (CT scan) for assessment.<br>{extra}"""
    else:
        body = f"""<b style="color:#00FF88;">✓ No {DISEASE} Indicators Detected</b><br><br>
        Confidence: <b>{c:.1f}%</b>.<br><br>
        <b style="color:#00F5FF;">Note:</b><br>
        • A normal AI result does not definitively rule out disease.<br>
        • Clinical judgment and physician review remain essential.<br>
        • If symptoms persist, please consult a healthcare professional."""
    return f"""<div style="font-family:'Exo 2',sans-serif; font-size:0.88rem; line-height:1.8;
                color:var(--text);">{body}<br><br>{disclaimer}</div>"""


def tier_chip_html(res):
    return (f'<span class="tier-chip" style="color:{res["tier_color"]}; '
            f'border-color:{res["tier_color"]};">TRIAGE: {res["tier_label"]}</span>')


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">PNEUMOSCAN AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-sub">CHEST X-RAY TRIAGE HUB &nbsp;·&nbsp; ResNet18 &nbsp;·&nbsp; GRAD-CAM '
    '&nbsp;·&nbsp; MULTI-MODEL</div>',
    unsafe_allow_html=True,
)

tab_scan, tab_batch, tab_history, tab_about = st.tabs(
    ["🔬  SCAN", "📂  BATCH QUEUE", "🗂  HISTORY", "📋  ABOUT"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE SCAN
# ════════════════════════════════════════════════════════════════════════════
with tab_scan:
    uploaded_files = st.file_uploader(
        "Drop chest X-ray(s) here or click to browse — select multiple images at once",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="single_upload",
    )

    if not uploaded_files:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:var(--dim);">
            <div style="font-size:3rem; margin-bottom:1rem;">🫁</div>
            <div style="font-family:'Orbitron',monospace; font-size:0.75rem;
                        letter-spacing:4px; color:var(--dim);">
                AWAITING SCAN INPUT
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif model is None:
        st.error("Model not loaded. Place `pneumonia_model.pkl` in the project folder.")
    else:
        for idx, uf in enumerate(uploaded_files):
            st.markdown(f"""
            <div style="font-family:'Orbitron',monospace; font-size:0.65rem;
                        color:var(--cyan); letter-spacing:3px; margin:1.2rem 0 0.4rem;">
                ── SCAN: {uf.name} ──
            </div>
            """, unsafe_allow_html=True)

            raw_bytes     = uf.getvalue()
            raw_image     = Image.open(io.BytesIO(raw_bytes))
            display_image = apply_processing(raw_image, processing_mode)

            with st.spinner(f"Analyzing {uf.name}…"):
                res = analyze(raw_image, with_cam=show_cam)

            cam_caption = (f"Evidence for: {res['cam_class']} · warm colours = regions that influenced "
                           "the prediction (model attention, not a diagnosis)")
            w, h = raw_image.size
            info_html = f"""
            <div style="font-size:0.75rem; color:var(--dim); margin-top:0.5rem;
                        font-family:'Exo 2',sans-serif;">
                📄 {uf.name} &nbsp;|&nbsp; {w}×{h}px &nbsp;|&nbsp; {len(raw_bytes)/1024:.1f} KB
            </div>"""

            if view_mode == "Side by side":
                c1, c2 = st.columns(2, gap="large")
                c1.image(display_image, use_container_width=True, caption=f"Mode: {processing_mode}")
                if res["cam"] is not None:
                    c2.image(res["cam"], use_container_width=True, caption=cam_caption)
                else:
                    c2.info("Grad-CAM is turned off or unavailable for this model.")
                st.markdown(info_html, unsafe_allow_html=True)
                res_col = st.container()
            else:
                img_col, res_col = st.columns([1, 1], gap="large")
                with img_col:
                    if res["cam"] is None:
                        st.image(display_image, use_container_width=True,
                                 caption=f"Mode: {processing_mode}")
                    elif view_mode == "Tabs":
                        t_scan, t_cam = st.tabs(["🩻 Scan", "🔥 Grad-CAM"])
                        with t_scan:
                            st.image(display_image, use_container_width=True,
                                     caption=f"Mode: {processing_mode}")
                        with t_cam:
                            st.image(res["cam"], use_container_width=True, caption=cam_caption)
                    else:   # Stacked
                        st.image(display_image, use_container_width=True,
                                 caption=f"Mode: {processing_mode}")
                        st.image(res["cam"], use_container_width=True, caption=cam_caption)
                    st.markdown(info_html, unsafe_allow_html=True)

            # ── Results column ──
            with res_col:
                st.markdown(banner_html(res), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-badge">
                        <span class="metric-label">CONFIDENCE</span>
                        <span class="metric-value">{res['conf']:.1f}%</span>
                    </div>
                    <div class="metric-badge">
                        <span class="metric-label">PREDICTION</span>
                        <span class="metric-value" style="color:{'#FF3860' if res['is_pos'] else '#00FF88'};">
                            {res['label']}
                        </span>
                    </div>
                    <div class="metric-badge">
                        <span class="metric-label">SEVERITY</span>
                        <span class="metric-value" style="color:{res['sev_color']};">{res['sev_label']}</span>
                    </div>
                </div>
                <div style="margin-bottom:0.6rem;">{tier_chip_html(res)}</div>
                """, unsafe_allow_html=True)

                bars = "".join(
                    prob_bar_html(c, res["probs"][c], "prob-fill-r" if c == POS else "prob-fill-g")
                    for c in class_names
                )
                st.markdown(bars, unsafe_allow_html=True)
                st.markdown(severity_meter_html(res["pos_prob"], res["sev_label"], res["sev_color"]),
                            unsafe_allow_html=True)

                with st.expander(f"📋  CLINICAL NOTES — {uf.name}"):
                    st.markdown(notes_html(res), unsafe_allow_html=True)

                # ── English PDF report ──
                try:
                    pdf_bytes = build_report_pdf(
                        patient_id=patient_id, filename=uf.name, disease=DISEASE,
                        model_name=active_name, res=res,
                        original=raw_image, heatmap=res["cam"],
                    )
                    st.download_button(
                        "⬇  Download Report (PDF)", data=pdf_bytes,
                        file_name=f"report_{Path(uf.name).stem}.pdf", mime="application/pdf",
                        key=f"pdf_{idx}_{uf.name}",
                    )
                except Exception as e:
                    st.caption(f"PDF report unavailable ({e}). Try: pip install reportlab")

                # ── Patient history ──
                if patient_id:
                    add_scan(patient_id, active_name, uf.name, raw_bytes, res["label"],
                             res["conf"], res["pos_prob"], res["sev_label"], res["tier_label"])
                    st.caption(f"🗂 Saved to history for patient {patient_id}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PROCESSING + PRIORITY QUEUE
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="scan-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:var(--cyan);
                letter-spacing:3px; margin-bottom:0.8rem;">
        📂 BATCH SCAN MODE
    </div>
    <div style="font-size:0.85rem; color:var(--dim); margin-bottom:1rem;">
        Upload multiple X-rays. They are ranked into a priority queue: URGENT first, then REVIEW (model not sure), then ROUTINE. Export the queue as CSV.
    </div>
    """, unsafe_allow_html=True)

    # ── Upload mode toggle ──
    upload_mode = st.radio(
        "Upload Mode",
        ["📄 Select Individual Files", "📁 Load from Folder Path"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    batch_files = []   # list of PIL Images with .name attribute

    # ── Helper to wrap a PIL image so it looks like an UploadedFile ──
    class _ImgFile:
        def __init__(self, name, pil_img):
            self.name    = name
            self._img    = pil_img

    if upload_mode == "📄 Select Individual Files":
        _uploaded = st.file_uploader(
            "Upload X-ray images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload_files",
        )
        if _uploaded:
            batch_files = [_ImgFile(f.name, Image.open(f)) for f in _uploaded]

    else:
        # ── Folder path mode ──
        st.markdown("""
        <div style="font-size:0.8rem; color:var(--dim); margin-bottom:0.6rem; line-height:1.7;">
            Paste the full path to a folder on your computer. All
            <b style="color:#00F5FF;">JPG / PNG</b> images inside will be loaded automatically.<br>
            <span style="color:#6C7086; font-size:0.73rem;">
                Example: <code style="color:#00F5FF;">C:\\Users\\you\\Desktop\\xrays</code>
                &nbsp;or&nbsp;
                <code style="color:#00F5FF;">/home/you/xrays</code>
            </span>
        </div>
        """, unsafe_allow_html=True)

        folder_path = st.text_input(
            "Folder path",
            placeholder=r"C:\Users\you\Desktop\xrays",
            label_visibility="collapsed",
            key="folder_path_input",
        )

        if folder_path:
            folder_path = folder_path.strip().strip('"').strip("'")  # handle quoted paths
            if not os.path.isdir(folder_path):
                st.error(f"❌ Folder not found: `{folder_path}`")
            else:
                exts = {".jpg", ".jpeg", ".png"}
                found = sorted([
                    f for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in exts
                ])
                if not found:
                    st.warning("⚠ No JPG/PNG images found in that folder.")
                else:
                    st.markdown(f"""
                    <div style="font-size:0.8rem; color:#00FF88; margin:0.4rem 0 0.8rem;">
                        ✓ Found <b>{len(found)}</b> image(s) in
                        <code style="color:#00F5FF;">{folder_path}</code>
                    </div>
                    """, unsafe_allow_html=True)

                    load_errors = []
                    for fname in found:
                        fpath = os.path.join(folder_path, fname)
                        try:
                            img = Image.open(fpath)
                            img.load()          # force decode now to catch corrupt files
                            batch_files.append(_ImgFile(fname, img))
                        except Exception as e:
                            load_errors.append(fname)

                    if load_errors:
                        st.warning(f"Skipped {len(load_errors)} unreadable file(s): {', '.join(load_errors)}")

    if batch_files:
        if model is None:
            st.error("Model not loaded. Place `pneumonia_model.pkl` in the project folder.")
        else:
            results  = []
            prog_bar = st.progress(0, text="Processing scans…")
            for i, f in enumerate(batch_files):
                r = analyze(f._img, with_cam=False)
                r.update({"file": f.name, "image": f._img})
                results.append(r)
                prog_bar.progress((i + 1) / len(batch_files),
                                  text=f"Processed {i+1}/{len(batch_files)} scans…")
            prog_bar.empty()

            # ── Priority queue: URGENT → REVIEW → ROUTINE ──
            results = priority_sort(results)
            for rank, r in enumerate(results, start=1):
                r["rank"] = rank

            n_urgent  = sum(1 for r in results if r["tier"] == 0)
            n_review  = sum(1 for r in results if r["tier"] == 1)
            n_routine = sum(1 for r in results if r["tier"] == 2)

            st.markdown(f"""
            <div class="summary-row">
                <div class="summary-badge">
                    <div class="summary-num" style="color:var(--cyan);">{len(results)}</div>
                    <div class="summary-lbl">TOTAL</div>
                </div>
                <div class="summary-badge">
                    <div class="summary-num" style="color:#FF3860;">{n_urgent}</div>
                    <div class="summary-lbl">URGENT</div>
                </div>
                <div class="summary-badge">
                    <div class="summary-num" style="color:#FFD700;">{n_review}</div>
                    <div class="summary-lbl">REVIEW</div>
                </div>
                <div class="summary-badge">
                    <div class="summary-num" style="color:#00FF88;">{n_routine}</div>
                    <div class="summary-lbl">ROUTINE</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            pos_col = f"{POS.lower()}_prob"
            table_rows = [{
                "rank": r["rank"], "file": r["file"], "triage": r["tier_label"],
                "prediction": r["label"], "confidence": round(r["conf"], 2),
                pos_col: round(r["pos_prob"], 2), "severity": r["sev_label"],
                "not_sure": r["uncertain"],
            } for r in results]
            df_queue = pd.DataFrame(table_rows)

            st.markdown('<div class="sidebar-title">🚦 PRIORITY QUEUE (see patients in this order)</div>',
                        unsafe_allow_html=True)
            st.markdown(html_table(df_queue), unsafe_allow_html=True)

            # ── On-demand Grad-CAM for one scan ──
            if show_cam:
                options = [f"#{r['rank']}  {r['file']}" for r in results]
                pick = st.selectbox("🔥 Inspect Grad-CAM for a scan", options, key="batch_cam_pick")
                r = results[options.index(pick)]
                cam = make_cam(r["image"].convert("RGB"), r["label"])
                if cam is None:
                    st.caption("Grad-CAM unavailable for this model.")
                else:
                    c1, c2 = st.columns(2)
                    c1.image(r["image"], caption="Original", use_container_width=True)
                    c2.image(cam, caption="Grad-CAM", use_container_width=True)

            # ── Image grid in priority order ──
            cols = st.columns(3)
            for idx, r in enumerate(results):
                with cols[idx % 3]:
                    st.image(r["image"], use_container_width=True)
                    st.markdown(f"""
                    <div class="batch-card" style="border-top:3px solid {r['tier_color']};">
                        <div style="font-family:'Orbitron',monospace; font-size:0.65rem;
                                    color:{r['tier_color']}; letter-spacing:2px;">
                            #{r['rank']} · {r['tier_label']}
                        </div>
                        <div style="font-size:0.75rem; color:var(--dim); margin-top:3px;">
                            {r['label']} {r['conf']:.1f}% &nbsp;|&nbsp;
                            <span style="color:{r['sev_color']};">{r['sev_label']}</span>
                        </div>
                        <div style="font-size:0.65rem; color:var(--dim); margin-top:2px;
                                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                            {r['file']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.download_button(
                label="⬇  Download Priority Queue (CSV)",
                data=df_queue.to_csv(index=False),
                file_name="pneumoscan_priority_queue.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PATIENT HISTORY
# ════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:var(--cyan);
                letter-spacing:3px; margin-bottom:0.6rem;">🗂 PATIENT HISTORY</div>
    <div style="font-size:0.85rem; color:var(--dim); margin-bottom:1rem;">
        Enter a Patient ID in the sidebar before scanning. Each scan is saved locally
        (SQLite) so you can follow a patient's progress over time.
    </div>
    """, unsafe_allow_html=True)

    patients = list_patients()
    if not patients:
        st.info("No saved scans yet. Type a Patient ID in the sidebar, then scan an X-ray.")
    else:
        default_idx = patients.index(patient_id) if patient_id in patients else 0
        chosen = st.selectbox("Patient", patients, index=default_idx)
        df_hist = get_history(chosen)
        st.markdown(html_table(df_hist), unsafe_allow_html=True)

        model_choice = st.selectbox("Progress chart for model", sorted(df_hist["model_name"].unique()))
        df_m = df_hist[df_hist["model_name"] == model_choice].copy()
        if len(df_m) >= 2:
            df_m["scanned_at"] = pd.to_datetime(df_m["scanned_at"])
            st.line_chart(df_m.set_index("scanned_at")["pos_prob"], height=240)
            first, last = df_m["pos_prob"].iloc[0], df_m["pos_prob"].iloc[-1]
            direction = ("rising ↑ (worse)" if last > first + 5
                         else "falling ↓ (better)" if last < first - 5 else "about the same")
            st.caption(f"Disease probability went from {first:.1f}% to {last:.1f}% — {direction}. "
                       "Indicative only: it compares model outputs, not clinical findings.")
        else:
            st.caption("Scan this patient at least twice (different images) to see a progress chart.")

        with st.expander("🗑 Delete this patient's records"):
            if st.button(f"Delete all records for {chosen}", key="del_patient"):
                delete_patient(chosen)
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="scan-card">
            <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:var(--cyan);
                        letter-spacing:3px; margin-bottom:1rem;">🧠 MODEL ARCHITECTURE</div>
            <div style="font-size:0.88rem; line-height:1.9; color:var(--text);">
                <b style="color:#00F5FF;">Base Model:</b> ResNet18 (ImageNet pretrained)<br>
                <b style="color:#00F5FF;">Strategy:</b> Transfer Learning — frozen backbone<br>
                <b style="color:#00F5FF;">Custom Head:</b> Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→2)<br>
                <b style="color:#00F5FF;">Trainable Params:</b> ~66,000<br>
                <b style="color:#00F5FF;">Input:</b> 224×224 RGB (ImageNet normalization)<br>
                <b style="color:#00F5FF;">Output:</b> Softmax — NORMAL / PNEUMONIA<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="scan-card">
            <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:var(--cyan);
                        letter-spacing:3px; margin-bottom:1rem;">⚙ TRAINING SETUP</div>
            <div style="font-size:0.88rem; line-height:1.9; color:var(--text);">
                <b style="color:#00F5FF;">Dataset:</b> Kaggle Chest X-Ray (5,863 images)<br>
                <b style="color:#00F5FF;">Optimizer:</b> Adam (lr=1e-3, wd=1e-4)<br>
                <b style="color:#00F5FF;">Scheduler:</b> ReduceLROnPlateau<br>
                <b style="color:#00F5FF;">Imbalance Fix:</b> WeightedRandomSampler<br>
                <b style="color:#00F5FF;">Early Stopping:</b> Patience = 4<br>
                <b style="color:#00F5FF;">Platform:</b> Google Colab T4 GPU<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scan-card" style="border-top-color:#FF3860; margin-top:1rem;">
        <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:#FF3860;
                    letter-spacing:3px; margin-bottom:0.8rem;">⚠ MEDICAL DISCLAIMER</div>
        <div style="font-size:0.88rem; line-height:1.8; color:var(--text);">
            PneumoScan AI is developed strictly for <b>research and educational purposes</b>.<br>
            It is <b>NOT</b> a certified medical device and must <b>NOT</b> be used for clinical
            diagnosis or patient care decisions.<br>
            Always consult a licensed physician or radiologist for medical evaluation.<br><br>
            <span style="color:var(--dim);">
                <a href="https://github.com/Bisma6405/PneumonoScan_AI"
                   style="color:#00F5FF;">GitHub Repository</a>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    feat_col, lim_col = st.columns(2, gap="large")

    with feat_col:
        st.markdown("""
        <div class="scan-card" style="margin-top:1rem;">
            <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:var(--cyan);
                        letter-spacing:3px; margin-bottom:1rem;">✨ FEATURES</div>
            <div style="font-size:0.88rem; line-height:1.9; color:var(--text);">
                <b style="color:#00FF88;">✓</b> Single and batch X-ray analysis (JPG/PNG, many files or a folder)<br>
                <b style="color:#00FF88;">✓</b> Prediction with confidence, severity level and risk meter<br>
                <b style="color:#00FF88;">✓</b> Grad-CAM heatmap with adjustable settings<br>
                <b style="color:#00FF88;">✓</b> "Not sure" flag for low-confidence scans (refer to a doctor)<br>
                <b style="color:#00FF88;">✓</b> Priority queue: URGENT, REVIEW, ROUTINE, with CSV export<br>
                <b style="color:#00FF88;">✓</b> Patient history stored locally (SQLite) with a progress chart<br>
                <b style="color:#00FF88;">✓</b> English PDF report for each scan<br>
                <b style="color:#00FF88;">✓</b> Image enhancement modes (contrast, sharpen, edges, equalize)<br>
                <b style="color:#00FF88;">✓</b> Plug-in models: add a folder to <code>models/</code> for a new disease<br>
                <b style="color:#00FF88;">✓</b> 5 themes (incl. high contrast) and 3 scan views
            </div>
        </div>
        """, unsafe_allow_html=True)

    with lim_col:
        st.markdown("""
        <div class="scan-card" style="border-top-color:#FF8C00; margin-top:1rem;">
            <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:#FF8C00;
                        letter-spacing:3px; margin-bottom:1rem;">🚧 LIMITATIONS</div>
            <div style="font-size:0.88rem; line-height:1.9; color:var(--text);">
                <b style="color:#FF8C00;">•</b> Trained on one public Kaggle dataset (mostly paediatric X-rays); it may not
                generalize to other hospitals, scanners or adults<br>
                <b style="color:#FF8C00;">•</b> Only the classifier head was trained, so Grad-CAM is coarse and can
                highlight areas outside the lungs (edges, markers)<br>
                <b style="color:#FF8C00;">•</b> High confidence does not mean a correct result<br>
                <b style="color:#FF8C00;">•</b> The "not sure" flag is a simple confidence threshold, not a calibrated
                uncertainty estimate<br>
                <b style="color:#FF8C00;">•</b> Two classes only (normal vs pneumonia); no bacterial or viral distinction<br>
                <b style="color:#FF8C00;">•</b> Patient history is local and single-user, with no login or encryption<br>
                <b style="color:#FF8C00;">•</b> Research prototype, not a medical device
            </div>
        </div>
        """, unsafe_allow_html=True)