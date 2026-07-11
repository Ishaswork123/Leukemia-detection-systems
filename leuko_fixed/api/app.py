import os

# Force legacy Keras 2 behavior for compatibility with this .h5 model
# and the Grad-CAM code below. Must be set before tensorflow is imported.
# os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
from dotenv import load_dotenv
import time
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import base64
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel




from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img
)

from groq import Groq

# NEW: report generation importscd "d:\leuko-fixed (1)\leuko_fixed\api"
from report_generator import ReportData, generate_report
from groq_summary import generate_ai_summary

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Correctly point to the .env file one folder up
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
env_path = root_dir / '.env'

load_dotenv(dotenv_path=env_path)

# SECURITY FIX: load from environment instead of hardcoding the key.
# Put GROQ_API_KEY=gsk_... in your .env file (one folder above api/).
# The key that was previously hardcoded here has been exposed and should
# be rotated in the Groq console — generate a new key and put ONLY the
# new one in .env.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    print(f"DEBUG: Active API Key starts with -> '{GROQ_API_KEY[:10]}...'. Total length: {len(GROQ_API_KEY)}")
else:
    print(" GROQ_API_KEY not found in environment / .env file.")

API_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = API_DIR.parent.absolute()

app = FastAPI(title="Leukemia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  FIX: Verify template directory exists before mounting
TEMPLATES_DIR = PROJECT_ROOT / "templates"
if not TEMPLATES_DIR.exists():
    print(f"  WARNING: Templates directory not found at {TEMPLATES_DIR}")
    print(f"   Creating it now...")
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# FIX: Use string path for StaticFiles to avoid issues
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
STATIC_IMAGES = STATIC_DIR / 'images'

MODEL_PATH = PROJECT_ROOT / 'model' / 'fine_tuned_densenet_leukemia (1).h5'

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# NEW: where generated PDF reports are saved
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CLIP Gatekeeper (local, no API call)
# ---------------------------------------------------------

clip_model = None
clip_processor = None

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Candidate labels. The "blood smear" labels are the accept class;
# everything else is what we reject against.
GATEKEEPER_LABELS = [
    "a microscopic image of a blood smear with blood cells",
    "a microscope slide of blood cells",
    "a photo of a chart or graph",
    "a photo of a document or text page",
    "a photo of a person or selfie",
    "a photo of a landscape or scenery",
    "a photo of an everyday object",
    "a screenshot of a website or app",
]

# Indices 0 and 1 are the "valid" classes; everything else is "invalid"
VALID_LABEL_INDICES = {0, 1}

# Minimum confidence the top "valid" label needs before we accept the image
GATEKEEPER_THRESHOLD = 0.35


def get_clip_model():
    """Lazily load CLIP model + processor (loaded once, reused for every request)."""
    global clip_model, clip_processor

    if clip_model is not None:
        return clip_model, clip_processor

    try:
        print(f"📦 Loading CLIP gatekeeper model ({CLIP_MODEL_NAME})...")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model.eval()
        print("✅ CLIP gatekeeper model loaded successfully!")
        return clip_model, clip_processor

    except Exception as e:
        print(f"❌ Error loading CLIP gatekeeper model: {e}")
        return None, None


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

model = None


def get_model():
    global model

    if model is not None:
        return model

    if not MODEL_PATH.exists():
        print(f" Model not found: {MODEL_PATH}")
        return None

    try:
        print(f" Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )
        print(" Model loaded successfully!")
        return model

    except Exception as e:
        print(f" Error loading model: {e}")
        return None


# ---------------------------------------------------------
# Groq Client
# ---------------------------------------------------------

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print(" Groq client initialized successfully!")
    except Exception as e:
        print(f" Groq initialization failed: {e}")
        groq_client = None
else:
    print("  GROQ_API_KEY environment variable not set. Chat endpoint will be unavailable.")


CLASS_NAMES = [
    'Acute Lymphoblastic Leukemia (ALL)',
    'Acute Myeloid Leukemia (AML)',
    'Chronic Lymphocytic Leukemia (CLL)',
    'Chronic Myeloid Leukemia (CML)',
    'Healthy / Normal'
]

IMG_SIZE = 224


# ---------------------------------------------------------
# GradCAM Functions
# ---------------------------------------------------------

def find_last_conv_layer(model):
    """Find the last convolutional layer in the model."""

    if hasattr(model, "layers") and hasattr(model.layers[0], "layers"):
        search_layers = model.layers[0].layers
    else:
        search_layers = model.layers

    for layer in reversed(search_layers):
        if len(layer.output.shape) == 4:
            return layer.name

    raise ValueError(" No convolutional layer found in model")


def make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name,
        pred_index=None
):
    """Generate GradCAM heatmap for model interpretability."""

    grad_model = tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]
        if isinstance(conv_outputs, list):
            conv_outputs = conv_outputs[0]

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)

    return heatmap.numpy()
def analyze_heatmap_regions(heatmap_gray_u8, threshold=0.6):
    """
    Analyze the grayscale (uint8) Grad-CAM heatmap and estimate
    suspicious regions. Expects a single-channel uint8 array.
    """
    _, thresh = cv2.threshold(
        heatmap_gray_u8,
        int(threshold * 255),
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    abnormal_regions = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            abnormal_regions += 1

    return abnormal_regions, contours
# 2. Change the call site inside generate_and_save_gradcam()


def generate_and_save_gradcam(
        image_path,
        model,
        class_index,
        save_path
):
    """Generate and save Grad-CAM visualization.

    Returns:
        (success: bool, abnormal_count: int)
    """

    try:
        img = cv2.imread(str(image_path))

        if img is None:
            print(f" Cannot read image: {image_path}")
            return False, 0

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)

        last_layer = find_last_conv_layer(model)

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            last_layer,
            class_index
        )

        # Resize heatmap (still float [0,1]) to original image size
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Keep a grayscale uint8 copy BEFORE colorizing, for region analysis
        heatmap_gray_u8 = np.uint8(255 * heatmap_resized)

        # Colorized version for visualization only
        heatmap_color = cv2.applyColorMap(heatmap_gray_u8, cv2.COLORMAP_JET)

        output = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

        # Analyze on the grayscale heatmap, not the colorized one
        abnormal_count, contours = analyze_heatmap_regions(heatmap_gray_u8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    output, "Abnormal", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )

        cv2.imwrite(str(save_path), output)
        print(f" Grad-CAM saved to {save_path}")
        print(f"Suspicious regions: {abnormal_count}")

        return True, abnormal_count

    except Exception as e:
        print(f" Grad-CAM generation failed: {e}")
        return False, 0


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_with_tta(image_path, tta_steps=5):
    """
    Predict with Test-Time Augmentation (TTA) for better confidence.
    """

    loaded_model = get_model()

    if loaded_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Check model path and try again."
        )

    try:
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )

        preds = []

        for _ in range(tta_steps):
            batch = next(datagen.flow(img_array, batch_size=1))
            preds.append(loaded_model.predict(batch, verbose=0))

        avg_pred = np.mean(preds, axis=0)[0]
        class_idx = int(np.argmax(avg_pred))

        return (
            CLASS_NAMES[class_idx],
            float(avg_pred[class_idx]),
            class_idx
        )

    except Exception as e:
        print(f" Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the home page."""
    try:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={}
        )
    except Exception as e:
        print(f" Template error: {e}")
        return HTMLResponse(
            f"<h1>Error loading template</h1><p>Template directory: {TEMPLATES_DIR}</p><p>Error: {str(e)}</p>",
            status_code=500
        )


def is_valid_blood_sample(image_path: str) -> bool:
    """
    Uses a local CLIP model to check whether the uploaded image looks like
    a microscopic blood smear/sample, via zero-shot text-label similarity.
    No external API call, no network dependency.
    """
    loaded_clip_model, loaded_clip_processor = get_clip_model()

    if loaded_clip_model is None or loaded_clip_processor is None:
        print("CLIP gatekeeper not available. Skipping safety check.")
        # NOTE: fail-open. Consider returning False here instead if you'd
        # rather block uploads entirely when the gatekeeper can't run.
        return True

    try:
        image = Image.open(image_path).convert("RGB")

        inputs = loaded_clip_processor(
            text=GATEKEEPER_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = loaded_clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape: [1, num_labels]
            probs = logits_per_image.softmax(dim=1)[0]  # shape: [num_labels]

        top_idx = int(torch.argmax(probs).item())
        top_label = GATEKEEPER_LABELS[top_idx]
        top_score = float(probs[top_idx].item())

        is_valid = top_idx in VALID_LABEL_INDICES and top_score >= GATEKEEPER_THRESHOLD

        print(f"🔍 CLIP Gatekeeper Verdict: '{top_label}' "
              f"(score={top_score:.3f}) -> {'VALID' if is_valid else 'REJECTED'}")

        return is_valid

    except Exception as e:
        print(f" Gatekeeper verification failed due to error: {e}")
        # Same fail-open tradeoff as above — change to False if you want
        # uploads blocked when something goes wrong reading the image.
        return True


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict leukemia type from uploaded blood smear image.
    Returns: prediction label, confidence, abnormal region count, and Grad-CAM heatmap.
    """

    if not file:
        raise HTTPException(400, "No file uploaded")

    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(400, "Only JPG/PNG images are supported")

    tmp_path = UPLOAD_FOLDER / f"{int(time.time())}_{file.filename}"

    try:
        # Save uploaded file
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        # GATEKEEPER VALIDATION STEP
        if not is_valid_blood_sample(str(tmp_path)):
            raise HTTPException(
                status_code=400,
                detail="Invalid image. Please upload a valid microscopic blood sample or smear image."
            )

        # Make prediction (only runs if the gatekeeper approved the image)
        label, conf, class_idx = predict_with_tta(str(tmp_path))

        # Generate Grad-CAM heatmap
        heatmap_url = None
        abnormal_count = 0
        loaded_model = get_model()

        if loaded_model:
            STATIC_IMAGES.mkdir(parents=True, exist_ok=True)

            heatmap_name = f"heatmap_{int(time.time())}.jpg"
            heatmap_path = STATIC_IMAGES / heatmap_name

            gradcam_ok, abnormal_count = generate_and_save_gradcam(
                str(tmp_path),
                loaded_model,
                class_idx,
                str(heatmap_path)
            )
            if gradcam_ok:
                heatmap_url = f"/static/images/{heatmap_name}"

        return {
            "prediction": label,
            "confidence": round(conf, 4),
            "heatmap_url": heatmap_url,
            "abnormal_region_count": abnormal_count,
            "status": "success"
        }

    except HTTPException:
        raise  # Pass through planned HTTP exceptions directly to the user
    except Exception as e:
        print(f" Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
                print(f"Cleaned up: {tmp_path}")
            except Exception as e:
                print(f"Could not delete temp file {tmp_path}: {e}")


@app.post("/chat")
async def chat(data: dict):
    """
    Chat with Groq LLM for medical queries about leukemia.
    """

    if "message" not in data:
        raise HTTPException(400, "Invalid payload: 'message' field required")

    if not groq_client:
        raise HTTPException(
            503,
            "Groq API not configured. Set GROQ_API_KEY environment variable."
        )

    try:
        message = data.get("message", "").strip()
        if not message:
            raise HTTPException(400, "Message cannot be empty")

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional medical assistant specialized in Leukemia. "
                        "Provide accurate, evidence-based information and always recommend "
                        "consulting with healthcare professionals for medical decisions."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            max_tokens=1024
        )

        return {
            "response": completion.choices[0].message.content,
            "status": "success"
        }

    except Exception as e:
        print(f" Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Network or API error: {str(e)}"
        )


# ---------------------------------------------------------
# NEW: PDF Report Generation
# ---------------------------------------------------------

class ReportRequest(BaseModel):
    """Body for POST /report. Frontend sends back what /predict returned."""
    prediction: str
    confidence: float
    heatmap_url: Optional[str] = None       # was: str  (rejected null -> 422)
    abnormal_region_count: int = 0
    patient_id: Optional[str] = None        # was missing entirely -> AttributeError


@app.post("/report")
async def report(payload: ReportRequest):
    """
    Generate a PDF report from an existing prediction result.
    Call /predict first, then POST its returned values here.
    """
    print(payload.model_dump())

    # This check now actually gets reached for normal/negative scans
    if not payload.heatmap_url:
        raise HTTPException(400, "Heatmap URL is required to generate a report")

    heatmap_filename = payload.heatmap_url.split("/static/images/")[-1]
    heatmap_path = STATIC_IMAGES / heatmap_filename

    if not heatmap_path.exists():
        raise HTTPException(404, f"Heatmap image not found: {heatmap_path}")

    ai_summary = generate_ai_summary(
        groq_client=groq_client,
        prediction_label=payload.prediction,
        confidence=payload.confidence,
        abnormal_region_count=payload.abnormal_region_count,
    )

    report_data = ReportData(
        prediction_label=payload.prediction,
        confidence_score=payload.confidence,
        heatmap_path=str(heatmap_path),
        abnormal_region_count=payload.abnormal_region_count,
        ai_summary=ai_summary,
        patient_id=payload.patient_id,
    )

    output_path = REPORTS_DIR / f"report_{report_data.report_id}.pdf"
    generate_report(report_data, str(output_path))

    return {
        "report_id": report_data.report_id,
        "file_path": f"/download-report/{report_data.report_id}",
        "status": "success",
        "message": "Report generated successfully"
    }

@app.get("/download-report/{report_id}")
async def download_report(report_id: str):
    """
    Download a previously generated PDF report by report_id.
    """
    report_path = REPORTS_DIR / f"report_{report_id}.pdf"
    
    if not report_path.exists():
        raise HTTPException(404, f"Report not found: {report_id}")
    
    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=f"leukemia_report_{report_id}.pdf",
    )


@app.get("/status")
async def status():
    """Health check endpoint."""

    return {
        "deployment": "FastAPI",
        "backend": "FastAPI-CORS",
        "ai_model": "DenseNet-Leukemia-Inference",
        "model_loaded": get_model() is not None,
        "model_path": str(MODEL_PATH),
        "device": "CPU",
        "groq_active": groq_client is not None,
        "templates_dir": str(TEMPLATES_DIR),
        "templates_exist": TEMPLATES_DIR.exists()
    }


# ---------------------------------------------------------
# Startup Event
# ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize model and resources on startup."""
    print("\n" + "=" * 60)
    print("LEUKEMIA DETECTION API STARTUP")
    print("=" * 60)

    model = get_model()
    if model:
        print("Model loaded successfully")
    else:
        print("  Model not loaded - predictions will fail")

    print(f" Project root: {PROJECT_ROOT}")
    print(f"Templates: {TEMPLATES_DIR} (exists: {TEMPLATES_DIR.exists()})")
    print(f" Uploads: {UPLOAD_FOLDER}")
    print(f" Static: {STATIC_DIR} (exists: {STATIC_DIR.exists()})")
    print(f" Reports: {REPORTS_DIR} (exists: {REPORTS_DIR.exists()})")
    print(f" Groq API: {'Configured' if groq_client else 'NOT configured'}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # For production/remote access, change to 0.0.0.0

    uvicorn.run(
        "app:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 for Windows compatibility
        port=8002,
        reload=True
    )
