<div align="center">

<!-- 🖼️ PLACEHOLDER: replace with your logo — e.g. docs/assets/logo.png -->
<img src="docs/assets/logo.png" alt="LeukoScan Logo" width="140"/>

# 🧬 LeukoScan
### AI-Powered Leukemia Subtype Detection from Blood Smear Images

**Deep learning–assisted screening with explainable AI and clinical-style PDF reporting.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/your-username/leukoscan?style=for-the-badge&color=gold)](https://github.com/your-username/leukoscan/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/your-username/leukoscan?style=for-the-badge)](https://github.com/your-username/leukoscan/commits/main)

Upload a blood smear image → get a subtype prediction, a Grad-CAM attention map, and a downloadable AI-generated clinical report — in seconds.

<!-- 🎬 PLACEHOLDER: animated GIF of the app in action -->
<img src="docs/assets/demo.gif" alt="LeukoScan Demo" width="800"/>

[▶️ Watch Loom Demo](https://www.loom.com/share/your-demo-link) &nbsp;•&nbsp; [🌐 Live Demo](https://leukoscan.vercel.app) &nbsp;•&nbsp; [📄 Sample Report](docs/assets/sample_report.pdf) &nbsp;•&nbsp; [🐛 Report Bug](https://github.com/your-username/leukoscan/issues) &nbsp;•&nbsp; [✨ Request Feature](https://github.com/your-username/leukoscan/issues)

</div>

---

> [!WARNING]
> **LeukoScan is a research / portfolio project, not a certified medical device.** Every prediction and generated report is explicitly labeled as AI-assisted and requires confirmation by a licensed hematologist or oncologist before any clinical decision is made. Do not use this system for real diagnostic purposes.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Screenshots](#-screenshots)
- [Project Architecture](#-project-architecture)
- [Folder Structure](#-folder-structure)
- [AI Pipeline](#-ai-pipeline)
- [Model Details](#-model-details)
- [Explainable AI — Grad-CAM](#-explainable-ai--grad-cam)
- [AI-Generated PDF Report](#-ai-generated-pdf-report)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [API Endpoints](#-api-endpoints)
- [Technologies Used](#-technologies-used)
- [Performance](#-performance)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔬 Overview

**Leukemia** is a group of blood cancers that begin in the bone marrow and result in abnormal white blood cell production. Diagnosis and subtyping — distinguishing between **Acute Lymphoblastic Leukemia (ALL)**, **Acute Myeloid Leukemia (AML)**, **Chronic Lymphocytic Leukemia (CLL)**, and **Chronic Myeloid Leukemia (CML)** — traditionally relies on manual microscopic examination of blood smears by trained hematologists. This process is time-intensive, requires significant expertise, and is subject to inter-observer variability, especially in regions with limited access to specialist pathologists.

**Why AI can help:** Convolutional neural networks have demonstrated strong performance at recognizing the subtle morphological patterns — cell size, nuclear shape, chromatin texture — that distinguish leukemic cell lines from healthy ones. Used as a **decision-support tool** (not a replacement for a pathologist), a model like this can:

- Flag likely-abnormal samples faster, helping triage cases that need urgent specialist review
- Provide a second, consistent opinion to reduce human error and fatigue-driven misreads
- Make first-pass screening more accessible in settings with limited specialist availability

**Motivation:** LeukoScan was built to demonstrate an end-to-end, production-style ML system — not just a Jupyter notebook model, but a real deployed pipeline covering input validation, inference, explainability, LLM-generated clinical narrative, and automated PDF reporting, with the safety guardrails (disclaimers, hedged language, confidence thresholds) a real medical-adjacent tool needs.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Drag & Drop Upload** | Modern drop-zone interface for JPG/PNG blood smear images |
| 🧠 **AI Subtype Prediction** | DenseNet121 classifies the image into one of 5 classes (ALL, AML, CLL, CML, Healthy) |
| 🛡️ **Input Gatekeeper** | A local CLIP zero-shot classifier rejects non-blood-smear images *before* they reach the diagnostic model |
| 🔁 **Test-Time Augmentation (TTA)** | Predictions are averaged over multiple augmented passes for more stable confidence scores |
| 📊 **Confidence Score** | Calibrated probability returned alongside every prediction |
| 🔥 **Grad-CAM Heatmap** | Visual attention map showing which regions of the image drove the prediction |
| 🚩 **Abnormal Region Detection** | Contour analysis over the heatmap counts and boxes suspicious regions |
| 🤖 **AI Clinical Summary** | Groq-hosted LLaMA 3.3 70B writes a hedged, disclaimer-safe narrative summary of each result |
| 💬 **In-App Medical Chatbot** | Ask general leukemia-related questions through a chat widget backed by the same LLM |
| 📄 **Downloadable PDF Report** | One-click clinical-style report combining prediction, heatmap, and AI summary |
| ⚡ **Async FastAPI Backend** | Fully asynchronous REST API with CORS enabled for external clients |
| 🎨 **Responsive UI** | Clean HTML/CSS/JS frontend with loading states and inline error handling |

---

## 🎥 Demo

<div align="center">

|  |  |
|---|---|
| 🎬 **Loom Walkthrough** | <!-- PLACEHOLDER --> [Watch the full walkthrough →](https://www.loom.com/share/98c9d937333344e0a4f7a4018b3366ee) |
| 🌐 **Live Website** | <!-- PLACEHOLDER --> [netfliy](https://leukoscan.vercel.app) |
| 🖼️ **Screenshots** | See [Screenshots](#-screenshots) below |

</div>

---

## 📸 Screenshots

<details open>
<summary><strong>Click to expand / collapse gallery</strong></summary>

<br>

**Homepage & Upload**
<!-- PLACEHOLDER: docs/assets/screenshot-home.png -->
<img src="\static\images\dashboard.jpeg" width="800"/>

**Prediction Result**
<!-- PLACEHOLDER: docs/assets/screenshot-prediction.png -->
<img src="\static\images\report.jpeg" width="800"/>

**Grad-CAM Visualization**
<!-- PLACEHOLDER: docs/assets/screenshot-gradcam.png -->
<img src="\static\images\gradcam.jpeg" width="800"/>

**Generated PDF Report**
<!-- PLACEHOLDER: docs/assets/screenshot-report.png -->
<img src="\static\images\ai_report.jpeg" width="800"/>

**Medical Chatbot**
<!-- PLACEHOLDER: docs/assets/screenshot-chatbot.png -->
<img src="\static\images\chatbot.jpeg" width="800"/>

</details>

---

## 🏗️ Project Architecture

LeukoScan follows a straightforward client → API → ML pipeline architecture, kept intentionally simple so every stage is inspectable and testable in isolation:

```
┌──────────────────┐      HTTP (multipart/JSON)     ┌───────────────────────┐
│   Browser (UI)    │ ─────────────────────────────▶ │      FastAPI App       │
│ HTML / CSS / JS   │ ◀───────────────────────────── │      (api/app.py)      │
└──────────────────┘                                 └───────────┬───────────┘
                                                                    │
                     ┌───────────────────────┬───────────────────┼───────────────────────┐
                     ▼                       ▼                    ▼                       ▼
           ┌──────────────────┐   ┌──────────────────┐  ┌──────────────────┐   ┌──────────────────┐
           │  CLIP Gatekeeper  │   │  DenseNet121 CNN  │  │   Grad-CAM Engine │   │   Groq LLM Client │
           │  (input validity) │   │  (+ TTA inference) │  │  (OpenCV overlay) │   │ (summary + chat)  │
           └──────────────────┘   └──────────────────┘  └──────────────────┘   └──────────────────┘
                                                                    │
                                                                    ▼
                                                        ┌──────────────────────┐
                                                        │  ReportLab PDF Engine │
                                                        │  (api/report_generator)│
                                                        └──────────────────────┘
```

**Design decisions worth noting:**
- The CLIP gatekeeper runs **locally with zero-shot text labels**, so invalid uploads (selfies, screenshots, random photos) are rejected before ever reaching the diagnostic model — no wasted inference, no misleading predictions on garbage input.
- Uploaded images are deleted immediately after inference (`finally` cleanup in `/predict`); only the derived Grad-CAM heatmap persists, so PDF reports are generated from the heatmap and stored metadata rather than the original patient image.
- The Groq LLM is prompted with strict system rules (hedged language, mandatory disclaimer, low-confidence flagging) so AI-generated report text never asserts a definitive diagnosis.

---

## 📁 Folder Structure

```
leuko_fixed/
├── api/
│   ├── app.py                 # FastAPI app: routes, model loading, Grad-CAM, gatekeeper
│   ├── groq_summary.py        # LLM prompt logic for the AI report summary
│   └── report_generator.py    # ReportLab PDF report builder
├── model/
│   └── fine_tuned_densenet_leukemia.h5   # Trained DenseNet121 weights
├── static/
│   ├── css/style.css          # App styling
│   ├── js/main.js             # Upload, drag-and-drop, prediction UI logic
│   ├── js/report.js           # Report generation/download logic
│   └── images/                # Generated Grad-CAM heatmaps
├── templates/
│   └── index.html             # Single-page frontend (upload UI + chatbot)
├── reports/                   # Generated PDF reports
├── uploads/                   # Temporary upload staging (cleared after inference)
├── requirements.txt
├── vercel.json                # Vercel serverless deployment config
└── .env                       # GROQ_API_KEY (not committed — see Installation)
```

---

## 🔄 AI Pipeline

```
   Image Upload
        │
        ▼
 CLIP Gatekeeper (reject non-blood-smear images)
        │
        ▼
   Preprocessing (resize 224×224, normalize)
        │
        ▼
 DenseNet121 + Test-Time Augmentation
        │
        ▼
 Prediction + Confidence Score
        │
        ▼
  Grad-CAM Heatmap + Abnormal Region Count
        │
        ▼
  Groq LLM Clinical Summary
        │
        ▼
    PDF Report (ReportLab)
```

---

## 🧠 Model Details

| Attribute | Value |
|---|---|
| **Architecture** | DenseNet121 (transfer learning, fine-tuned) |
| **Framework** | TensorFlow / Keras (legacy Keras 2 mode) |
| **Input Size** | 224 × 224 × 3 |
| **Output Classes** | 5 |
| **Classes** | ALL, AML, CLL, CML, Healthy / Normal |
| **Inference Strategy** | Test-Time Augmentation (5-step: rotation, shift, flip, zoom) |
| **Explainability** | Grad-CAM on the last convolutional layer |

> [!NOTE]
> Training-time metrics (accuracy, precision, recall, F1, confusion matrix, ROC curves) belong to the model training pipeline/notebook, not this inference repo. Add your own numbers and plots below once you have them documented.

<div align="center">

| Metric | Score |
|---|---|
| Accuracy | `TODO` |
| Precision | `TODO` |
| Recall | `TODO` |
| F1 Score | `TODO` |

<!-- PLACEHOLDER: docs/assets/confusion_matrix.png -->
<img src="\static\images\confusion.png" width="450"/>
<!-- PLACEHOLDER: docs/assets/roc_curve.png -->
<img src="\static\images\download.png" width="450"/>

</div>

---

## 🔍 Explainable AI — Grad-CAM

Deep learning models are often criticized as "black boxes," which is a serious liability in any medical-adjacent context. LeukoScan addresses this using **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

For each prediction, LeukoScan:
1. Finds the last convolutional layer of DenseNet121
2. Backpropagates the gradient of the predicted class through that layer
3. Pools the gradients to weight each feature map's contribution
4. Produces a heatmap highlighting the image regions most responsible for the prediction
5. Overlays the heatmap on the original image and draws bounding boxes around high-attention contour regions (`analyze_heatmap_regions`)

This gives clinicians a visual sanity check — if the model is "looking" at the actual cell morphology rather than image artifacts or background noise, that's a meaningful trust signal.

---

## 📄 AI-Generated PDF Report

Every report generated by `report_generator.py` includes:

- **Header** — clinic/report title, unique report ID, generation timestamp
- **Prediction Summary** — predicted subtype and confidence score, color-coded by confidence band
- **Grad-CAM Visualization** — the heatmap overlay image embedded directly in the PDF
- **Abnormal Region Count** — number of high-attention regions detected
- **AI Clinical Summary** — a 4–6 sentence, hedged-language paragraph generated by Groq's LLaMA 3.3 70B, explicitly avoiding definitive diagnostic claims and flagging low-confidence results as inconclusive
- **Medical Disclaimer** — a mandatory footer stating the report is AI-generated, for research/informational support only, and must be reviewed by a licensed hematologist or oncologist before any clinical decision

> [!TIP]
> The original uploaded image is intentionally **not** included in the PDF — the report is built only from the derived heatmap and known prediction values, since the raw upload is deleted right after inference.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip
- A [Groq API key](https://console.groq.com/) (free tier available)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/ishaswork123/leukoscan.git
cd leukoscan/leuko_fixed

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then edit .env and add your key:
# GROQ_API_KEY=gsk_your_key_here

# 5. Run the development server
cd api
uvicorn app:app --reload --port 8002
```

The app will be available at **http://127.0.0.1:8002**.

> [!WARNING]
> Never commit your `.env` file or hardcode API keys in source. If a key has ever been committed to git history, rotate it in the [Groq console](https://console.groq.com/keys) immediately, even after removing it from the file.

---

## 🚀 Usage

1. Open the app in your browser
2. **Drag & drop** (or click to browse) a blood smear microscopy image — JPG or PNG only
3. The CLIP gatekeeper validates the image is a genuine blood smear
4. Click **Analyze** — the model returns the predicted subtype, confidence score, and Grad-CAM heatmap
5. Review the AI attention map and flagged abnormal regions
6. Click **Generate Report** to produce a downloadable clinical-style PDF
7. Use the chatbot 💬 to ask general questions about leukemia

---

## ☁️ Deployment

LeukoScan is configured for **serverless deployment on Render** via `vercel.json`, which routes all incoming requests to the FastAPI ASGI app as a Python serverless function.

```bash
# Install the Vercel CLI
npm i -g vercel

# From the project root
vercel

# Set your environment variable in the Vercel dashboard
# Project → Settings → Environment Variables → GROQ_API_KEY
```

> [!NOTE]
> The bundled `.h5` model file is large; serverless platforms impose function size/cold-start limits. For production traffic, consider hosting the model separately (e.g. a dedicated inference service or container) and having the API call out to it, rather than bundling the weights into the serverless function.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/predict` | Accepts an image file, returns prediction, confidence, heatmap URL, and abnormal region count |
| `POST` | `/report` | Generates a PDF report from a prior `/predict` result |
| `GET` | `/download-report/{report_id}` | Downloads a previously generated PDF report |
| `POST` | `/chat` | Sends a message to the Groq-powered medical assistant chatbot |
| `GET` | `/status` | Health check — model load status, template/static dir status, Groq connectivity |

<details>
<summary><strong>Example: <code>POST /predict</code></strong></summary>

```bash
curl -X POST http://127.0.0.1:8002/predict \
  -F "file=@sample_smear.jpg"
```

```json
{
  "prediction": "Acute Lymphoblastic Leukemia (ALL)",
  "confidence": 0.9421,
  "heatmap_url": "/static/images/heatmap_1783696305.jpg",
  "abnormal_region_count": 3,
  "status": "success"
}
```
</details>

---

## 🛠️ Technologies Used

<div align="center">

| Category | Technologies |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | FastAPI, Uvicorn, Python |
| **Machine Learning** | TensorFlow, Keras (legacy mode), DenseNet121 |
| **Input Validation** | CLIP (`openai/clip-vit-base-patch32`) zero-shot gatekeeper |
| **Explainable AI** | Grad-CAM |
| **LLM / Generative AI** | Groq API — LLaMA 3.3 70B Versatile |
| **Image Processing** | OpenCV, Pillow, NumPy |
| **PDF Generation** | ReportLab |
| **Deployment** |Render (serverless) |
| **Version Control** | Git, GitHub |

</div>

---

## 📈 Performance

| Aspect | Notes |
|---|---|
| Inference | TTA (5 augmented passes) trades a small latency cost for more stable confidence scores |
| Gatekeeper overhead | CLIP model runs once per upload, lazily loaded and cached across requests |
| Cold starts | Serverless deployment may incur cold-start latency on first request due to TensorFlow + CLIP model load |
| Report generation | Typically a few seconds, dominated by the Groq LLM summary call |

<!-- PLACEHOLDER: replace with real benchmark numbers once you've measured them, e.g. p50/p95 latency, throughput -->

---

## 🔮 Future Improvements

- [ ] Persistent prediction history (database-backed)
- [ ] User authentication & role-based access (patient / clinician / admin)
- [ ] Doctor review dashboard for confirming or overriding AI predictions
- [ ] Cloud storage for uploads and reports (S3 / GCS)
- [ ] Multi-model ensemble support
- [ ] SHAP-based explanations alongside Grad-CAM
- [ ] Public, versioned REST API with rate limiting
- [ ] Dockerized deployment
- [ ] CI/CD pipeline (GitHub Actions) with automated tests

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please open an issue first for major changes to discuss what you'd like to change.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 📬 Contact

<div align="center">

**Isha Eman**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ishaswork1233)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/isha-eman-844313279/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:isha77477@gmail.com)

⭐️ If this project helped you, consider giving it a star!

</div>
