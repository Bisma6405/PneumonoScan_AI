"""
clinic_tools.py - extra features for PneumoScan AI (added during the hackathon)

1. Grad-CAM heatmaps        -> gradcam_overlay()
2. Uncertainty + triage     -> triage_tier(), priority_sort()
3. Patient history (SQLite) -> add_scan(), get_history(), list_patients(), delete_patient()
4. English PDF report       -> build_report_pdf()
5. Plug-in model registry   -> discover_models(), load_model_files()

License: MIT
"""
import hashlib
import io
import json
import os
import pickle
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# 1. Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_layer(model, path):
    """'layer4' or 'features.7.2' -> the actual submodule."""
    module = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _jet(x):
    """x in [0,1] (H,W) -> RGB (H,W,3) in 0..255, a jet-like colormap without matplotlib."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255


def gradcam_overlay(model, pil_rgb, transform, class_idx, layer_name="layer4",
                    alpha=0.65, threshold=0.35):
    """
    Grad-CAM: highlights the image regions that pushed the model towards `class_idx`.

    alpha      overall heat strength (0..1)
    threshold  fraction of the strongest activation below which nothing is coloured
               (0.35 = hide the weakest 35%; removes the blue "haze" over the whole image)
    Returns a PIL image (the X-ray with the heatmap on top).
    """
    layer = _resolve_layer(model, layer_name)
    store = {}

    def forward_hook(_module, _inp, out):
        store["act"] = out
        out.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        # The backbone is frozen, so the input must require grad for backward() to work.
        x = transform(pil_rgb).unsqueeze(0).requires_grad_(True)
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x)
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    act = store["act"].detach()[0]                      # (C, h, w)
    grad = store["grad"].detach()[0]                    # (C, h, w)
    weights = grad.mean(dim=(1, 2), keepdim=True)       # global-average-pool the gradients
    cam = torch.relu((weights * act).sum(dim=0))        # (h, w)

    # smooth upsampling on floats (no 8-bit blockiness) to the original image size
    W, H = pil_rgb.size
    cam = F.interpolate(cam[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # hide weak regions, stretch the rest back to 0..1
    mask = np.clip((cam - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)
    heat = _jet(mask)
    base = np.asarray(pil_rgb).astype(np.float32)
    w = mask[..., None] * alpha
    out = base * (1 - w) + heat * w
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Uncertainty + triage priority
# ══════════════════════════════════════════════════════════════════════════════
TIERS = {
    0: ("URGENT",  "#FF3860"),   # confident positive
    1: ("REVIEW",  "#FFD700"),   # model not sure -> show to a doctor
    2: ("ROUTINE", "#00FF88"),   # confident negative
}


def triage_tier(is_positive, uncertain):
    if uncertain:
        return 1
    return 0 if is_positive else 2


def priority_sort(results):
    """Most urgent first; inside a tier, the higher disease probability first."""
    return sorted(results, key=lambda r: (r["tier"], -r["pos_prob"]))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Patient history (SQLite)
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = os.environ.get("PNEUMOSCAN_DB", "pneumoscan_history.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS scans (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            scanned_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            filename   TEXT,
            file_hash  TEXT NOT NULL,
            prediction TEXT,
            confidence REAL,
            pos_prob   REAL,
            severity   TEXT,
            triage     TEXT,
            UNIQUE(patient_id, model_name, file_hash)
        )"""
    )
    return c


def add_scan(patient_id, model_name, filename, file_bytes,
             prediction, confidence, pos_prob, severity, triage):
    """Saves a scan. The same image for the same patient+model is stored only once."""
    file_hash = hashlib.md5(file_bytes).hexdigest()
    with closing(_conn()) as c:
        with c:
            cur = c.execute(
                """INSERT OR IGNORE INTO scans
                   (patient_id, scanned_at, model_name, filename, file_hash,
                    prediction, confidence, pos_prob, severity, triage)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (patient_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name,
                 filename, file_hash, prediction, round(confidence, 2),
                 round(pos_prob, 2), severity, triage),
            )
        return cur.rowcount == 1


def list_patients():
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT patient_id FROM scans GROUP BY patient_id ORDER BY MAX(scanned_at) DESC"
        ).fetchall()
    return [r[0] for r in rows]


def get_history(patient_id):
    with closing(_conn()) as c:
        return pd.read_sql_query(
            """SELECT scanned_at, model_name, filename, prediction, confidence,
                      pos_prob, severity, triage
               FROM scans WHERE patient_id = ? ORDER BY scanned_at""",
            c, params=(patient_id,),
        )


def delete_patient(patient_id):
    with closing(_conn()) as c:
        with c:
            c.execute("DELETE FROM scans WHERE patient_id = ?", (patient_id,))


# ══════════════════════════════════════════════════════════════════════════════
# 4. English PDF report
# ══════════════════════════════════════════════════════════════════════════════
def _pil_to_rl_image(pil_img, width_cm):
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage

    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
    buf.seek(0)
    w, h = pil_img.size
    width = width_cm * cm
    return RLImage(buf, width=width, height=width * h / w)


def build_report_pdf(patient_id, filename, disease, model_name, res, original, heatmap=None):
    """
    Returns PDF bytes: a short, plain-English report a patient can read.
    `res` is the dict produced by the app's analyze() function.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=15)
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#0D3B66"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#0D3B66"))
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"))

    # ── Wording ───────────────────────────────────────────────────────────────
    d = escape(disease)
    if res["uncertain"]:
        result_short = "Not sure - please see a doctor"
        verdict = "The model cannot say with enough confidence what this scan shows."
        advice = [
            "Please show this X-ray to a doctor or radiologist.",
            "Do not rely on this result alone.",
        ]
    elif res["is_pos"]:
        result_short = "Signs found"
        verdict = f"This X-ray shows signs similar to <b>{d}</b>."
        advice = [
            "See a doctor or radiologist as soon as possible.",
            "Seek care without delay if you have fever, cough or difficulty breathing.",
            "The doctor may ask for more tests (for example blood tests or a CT scan).",
        ]
    else:
        result_short = "No clear signs found"
        verdict = f"No clear signs of <b>{d}</b> were found on this X-ray."
        advice = [
            "A normal AI result does not completely rule out disease.",
            "If symptoms continue or get worse, please see a doctor.",
        ]

    severity = {"LOW": "Low", "MODERATE": "Moderate", "HIGH": "High",
                "CRITICAL": "Critical"}.get(res["sev_label"], res["sev_label"])

    story = [
        Paragraph("PneumoScan AI - Scan Report", h1),
        Paragraph("Summary for the patient (research prototype)",
                  ParagraphStyle("sub", parent=small, alignment=1)),
        Spacer(1, 0.4 * cm),
    ]

    rows = [
        ("Patient ID", patient_id or "-"),
        ("Date", datetime.now().strftime("%d-%m-%Y %H:%M")),
        ("File", filename),
        ("Model", model_name),
        ("Result", result_short),
        ("Model confidence", f"{res['conf']:.1f}%"),
        ("Severity estimate", severity),
    ]
    table = Table(
        [[Paragraph(f"<b>{escape(k)}</b>", body), Paragraph(escape(str(v)), body)] for k, v in rows],
        colWidths=[5.5 * cm, 11 * cm],
    )
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F1F8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [table, Spacer(1, 0.5 * cm), Paragraph("Result", h2), Paragraph(verdict, body)]

    story += [Spacer(1, 0.2 * cm), Paragraph("What you should do", h2)]
    story += [Paragraph(f"&bull; {a}", body) for a in advice]

    # Images (original + Grad-CAM)
    story += [Spacer(1, 0.4 * cm), Paragraph("X-ray", h2)]
    if heatmap is not None:
        img_table = Table(
            [[_pil_to_rl_image(original, 7.5), _pil_to_rl_image(heatmap, 7.5)],
             [Paragraph("Original X-ray", small),
              Paragraph("AI attention map (warm colours = areas that influenced the result)", small)]],
            colWidths=[8.2 * cm, 8.2 * cm],
        )
    else:
        img_table = Table([[_pil_to_rl_image(original, 7.5)], [Paragraph("Original X-ray", small)]])
    story.append(img_table)

    story += [
        Spacer(1, 0.6 * cm),
        Paragraph("Important note", h2),
        Paragraph(
            "This report is for research and educational purposes only. It is not a medical "
            "diagnosis or treatment advice. Please consult a licensed doctor before making "
            "any decision about your health.", body),
    ]

    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="PneumoScan AI Report").build(story)
    return out.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Plug-in model registry
# ══════════════════════════════════════════════════════════════════════════════
def discover_models(models_dir="models"):
    """
    Finds every available model:
      - the original pneumonia_model.pkl + model_metadata.json in the project root
      - any models/<folder>/model.pkl + metadata.json
    Returns {display_name: (model_path, metadata_path)}
    """
    found = {}
    if os.path.exists("pneumonia_model.pkl"):
        found["Pneumonia (default)"] = ("pneumonia_model.pkl", "model_metadata.json")

    base = Path(models_dir)
    if base.is_dir():
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            model_path, meta_path = folder / "model.pkl", folder / "metadata.json"
            if not model_path.exists():
                continue
            name = folder.name
            if meta_path.exists():
                try:
                    name = json.loads(meta_path.read_text()).get("display_name", folder.name)
                except Exception:
                    pass
            found[name] = (str(model_path), str(meta_path))
    return found


def load_model_files(model_path, meta_path):
    """
    Loads a pickled model + its metadata JSON.
    WARNING: pickle can run arbitrary code - only load model files you trust.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.eval()
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return model, meta