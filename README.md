# 🫁 PneumoScan AI — X-Ray Triage Hub

A deep learning-powered chest X-ray analysis web application built with **Streamlit** and **PyTorch**. It classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a ResNet18 model trained via transfer learning, and helps a busy clinic decide **who to look at first**: it explains each prediction with a heatmap, flags scans it is not sure about, ranks batches by urgency, keeps a patient history and produces a plain-English PDF report.

Built for **Hacktoberfest Hackathon NUST x NSTP**. Open source under the **MIT License**.

> ⚠️ Research and educational prototype. **Not** a medical device. See the [Disclaimer](#️-disclaimer).

---

## 🎯 Problem Statement

In small and under-resourced clinics, one radiologist or doctor may have to go through dozens of chest X-rays a day. Pneumonia needs fast attention, but:

- urgent cases are buried in the pile and wait their turn,
- AI tools usually give a bare label with no explanation, so doctors cannot tell whether to trust it,
- and a model that is wrong still sounds confident.

**Intended users:** doctors, radiologists and students in small clinics or teaching settings, and developers who want an open, inspectable baseline for X-ray triage.

## 💡 Our Solution

PneumoScan classifies each X-ray, then adds the pieces a real workflow needs: an explanation (Grad-CAM), a "not sure, show a doctor" flag, a priority queue, a patient history and a printable report. A plug-in system lets other disease models be added without touching the app code.

---

## 📸 Features

- **Single Scan** — Upload one or more X-rays and get an instant prediction with confidence score, probability bars, severity and a risk meter
- **Grad-CAM Heatmap** — See which regions influenced the prediction; adjustable class, strength, weak-area cut-off and detail level
- **Uncertainty Flag** — Scans below a confidence threshold (default 80%, adjustable) are marked **NOT SURE / REVIEW** so a doctor looks at them
- **Priority Queue (Batch Scan)** — Upload many X-rays; results are ranked **URGENT → REVIEW → ROUTINE** with a summary dashboard and CSV export
- **Patient History** — Enter a Patient ID and every scan is saved locally (SQLite); a progress chart shows how the disease probability changes over time
- **English PDF Report** — A short, plain-language report for each scan with result, advice, the X-ray and the heatmap
- **Plug-in Models** — Drop a folder into `models/` to add another disease model (for example TB); it appears in a dropdown
- **Image Processing Modes** — View scans in Original, High Contrast, Sharpen, Edge Highlight or Equalized mode
- **Severity Rating** — LOW / MODERATE / HIGH / CRITICAL classification based on disease probability
- **Clinical Notes** — Contextual interpretation and recommended next steps for each result
- **Themes and Views** — 5 themes (Neon Dark, Soft Dark, High Contrast Dark, Light, High Contrast Light) and 3 scan views (Tabs, Side by side, Stacked)
- **Model Info Sidebar** — Live display of test accuracy, AUC-ROC, epochs trained and system status

---

## 🚦 How Triage Works

| Tier | When | Meaning |
|---|---|---|
| 🔴 URGENT | Confident that the scan shows the disease | See this patient first |
| 🟡 REVIEW | Confidence is below the threshold | The model is not sure; a doctor must look |
| 🟢 ROUTINE | Confident that the scan looks normal | Lower priority |

Inside a tier, scans are sorted by disease probability (highest first).

> The "not sure" flag is a simple confidence cut-off. It is **not** a calibrated uncertainty estimate.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | ResNet18 (pretrained on ImageNet) |
| Strategy | Transfer learning — frozen backbone, custom head |
| Task | Binary Classification — NORMAL vs PNEUMONIA |
| Trainable Parameters | ~66K (custom head only) |
| Input Size | 224 × 224 RGB |
| Normalization | ImageNet mean/std |
| Output | Softmax probabilities for 2 classes |

### Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Regularization | Dropout(0.4) + Dropout(0.3) |
| Class Imbalance | WeightedRandomSampler |
| Early Stopping | Patience = 4 epochs on val loss |
| Mixed Precision | float16 (GPU only) |
| Max Epochs | 15 |

### Model Architecture (Custom Head)

```
ResNet18 (frozen backbone)
    └── Dropout(p=0.4)
    └── Linear(512 → 256)
    └── ReLU()
    └── Dropout(p=0.3)
    └── Linear(256 → 2)
```

### Performance

| Metric | Value |
|---|---|
| Test Accuracy | 93.63% |
| AUC-ROC | 0.9682 |

---

## 📁 Project Structure

```
PneumonoScan_AI/
├── app.py                    # Streamlit application (UI, tabs, themes)
├── clinic_tools.py           # Grad-CAM, triage, history DB, PDF report, model registry
├── evaluate.py               # Checks model accuracy on the test images
├── requirements.txt          # Python dependencies
├── pneumonia_model.pkl       # Trained default model (download from Colab)
├── model_metadata.json       # Class names, accuracy, input specs
├── training_history.pkl      # Loss/accuracy per epoch (optional)
├── models/                   # Optional plug-in models
│   └── <model-name>/
│       ├── model.pkl
│       └── metadata.json
├── pneumoscan_history.db     # Created automatically (local patient data, not committed)
├── LICENSE                   # MIT
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone / download the project

```bash
git clone https://github.com/Bisma6405/PneumonoScan_AI.git
cd PneumonoScan_AI
```

### 2. Create a virtual environment (Python 3.11 recommended)

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model files

Download the following files from your Google Colab training notebook and place them in the project root:

- `pneumonia_model.pkl`
- `model_metadata.json`
- `training_history.pkl` *(optional)*

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit>=1.35.0
torch>=2.2.0
torchvision>=0.17.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
reportlab>=4.0.0
```

---

## 🗂️ Dataset

The model was trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

```
chest_xray/
├── train/
│   ├── NORMAL/        (1,341 images)
│   └── PNEUMONIA/     (3,875 images)
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/        (234 images)
    └── PNEUMONIA/     (390 images)
```

> The training notebook fixes the tiny val set (16 images → ~312) by moving 50% of test images into val for stable validation curves.

---

## 🖥️ Usage Guide

### Single Scan (SCAN tab)

1. Open the app in your browser
2. Pick a theme and a scan view in the sidebar (optional)
3. Enter a **Patient ID** in the sidebar if you want the result saved to history
4. Upload a chest X-ray image (JPG/PNG)
5. Read the prediction, confidence, severity, triage tier and risk meter
6. Open the **Grad-CAM** view to see what the model focused on
7. Expand **Clinical Notes** for interpretation, or download the **PDF report**

### Batch Queue (BATCH QUEUE tab)

1. Switch to the **BATCH QUEUE** tab
2. Upload multiple chest X-ray images at once
3. The app ranks them: URGENT first, then REVIEW, then ROUTINE
4. Inspect the Grad-CAM of any scan from the dropdown
5. Click **Download Priority Queue (CSV)** to export results

### Patient History (HISTORY tab)

1. Scan X-rays with a Patient ID entered in the sidebar
2. Open the **HISTORY** tab and pick the patient
3. Review past results and the progress chart
4. Use the delete option to remove a patient's records

### Severity Levels

| Level | Disease Probability |
|---|---|
| 🟢 LOW | < 30% |
| 🟡 MODERATE | 30–60% |
| 🟠 HIGH | 60–80% |
| 🔴 CRITICAL | > 80% |

### Grad-CAM Settings

| Setting | What it does |
|---|---|
| Explain which class | Evidence for the predicted class, or always for the disease class |
| Hide weak areas (%) | Higher values colour only the strongest regions |
| Heat intensity | Strength of the colour overlay |
| Finer detail | Uses an earlier layer (sharper but noisier) |

---

## 🧩 Adding Another Model (Plug-ins)

1. Train a ResNet-style classifier in the same way as the default model and save it as a pickle
2. Create a folder `models/<your-model-name>/`
3. Put the model in `model.pkl` and add a `metadata.json`:

```json
{
  "display_name": "Tuberculosis",
  "disease_name": "Tuberculosis",
  "positive_class": "TUBERCULOSIS",
  "class_names": ["NORMAL", "TUBERCULOSIS"],
  "input_size": 224,
  "gradcam_layer": "layer4"
}
```

4. Restart the app. The model appears in the **Active model** dropdown.

> ⚠️ Model files are Python pickles, which can run code when loaded. Only load model files you trust.

---

## ✅ Checking Model Accuracy

`evaluate.py` scores the model on the test images and prints accuracy, precision, recall, F1, confusion matrix, sensitivity, specificity and AUC-ROC. It also shows how the "not sure" flag behaves on your data.

```bash
python evaluate.py --data "path/to/chest_xray/test"
```

| Option | What it does |
|---|---|
| `--split held-out` (default) | Scores only the test images that were **not** used for validation during training, which gives an honest result |
| `--split all` | Scores every image in the folder (slightly optimistic) |
| `--threshold 80` | Confidence cut-off used for the "not sure" report |
| `--out predictions.csv` | Saves every prediction to a CSV file |

---

## 🔄 Retraining the Model

Open the training notebook in Google Colab:

1. Go to **Runtime → Change runtime type → T4 GPU**
2. Mount your Google Drive containing the dataset
3. Run all cells — the notebook auto-detects the dataset path
4. Download `pneumonia_model.pkl`, `model_metadata.json` and `training_history.pkl`
5. Replace the files in your app folder

---

## 🚧 Limitations

- Trained on one public Kaggle dataset (mostly paediatric X-rays); it may not generalize to other hospitals, scanners or adults
- Only the classifier head was trained, so Grad-CAM is coarse and can highlight areas outside the lungs (edges, markers)
- High confidence does not mean a correct result
- The "not sure" flag is a simple confidence threshold, not a calibrated uncertainty estimate
- Two classes only (normal vs pneumonia); no bacterial or viral distinction
- Patient history is local and single-user, with no login or encryption
- Research prototype, not a medical device

### Future Work

- Fine-tune the last ResNet block and add border-cropping augmentation for better localization
- Calibrated uncertainty (for example conformal prediction)
- Add a TB model through the plug-in system
- Multi-class output and external validation on another hospital's data

---

## 🔒 Privacy and Responsible Use

- Patient data stays in a local SQLite file (`pneumoscan_history.db`) that is excluded from git. Use a code or ID, **not** a real name.
- The repository contains no API keys, passwords or tokens.
- Never upload real patient images to a public repository.

---

## ⚠️ Disclaimer

This application is intended **strictly for research and educational purposes**. It is **not** a certified medical device and should **not** be used for clinical diagnosis or patient care decisions. Always consult a licensed physician or radiologist for medical evaluation.

---

## 🛠️ Tech Stack

- [PyTorch](https://pytorch.org/) — model training and inference
- [Torchvision](https://pytorch.org/vision/) — ResNet18, transforms
- [Streamlit](https://streamlit.io/) — web application framework
- [Pillow](https://python-pillow.org/) — image loading and processing
- [pandas](https://pandas.pydata.org/) — tables and history queries
- [ReportLab](https://www.reportlab.com/) — PDF reports
- [SQLite](https://www.sqlite.org/) — local patient history
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics (AUC-ROC, confusion matrix)

---

## 📄 License

Released under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset by Paul Mooney on Kaggle
- ResNet18 weights from [Torchvision](https://pytorch.org/vision/) (ImageNet)
- Grad-CAM method: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Hacktoberfest Hackathon NUST x NSTP, organized by AWS Student Builder Group NUST

---

*Built using Transfer Learning on ResNet18*
