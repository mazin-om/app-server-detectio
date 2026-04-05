"""
Jetson Nano GPU Inference Server for Street Asset Detection.
FastAPI server that receives video frames from the Flutter app,
processes them with YOLO on GPU, and returns detection results.
"""

import os
import sys
import time
import math
import sys
import types
import time
import math
import logging
from datetime import datetime

# ─── CRITICAL HACK: Fix torchvision Runtime Error on Jetson ───
# The generic pip wheel crashes because it tries to register C++ ops 
# that don't exist in the system's libtorch. We suppress the registration
# module and then monkey-patch the NMS function with a pure Python version.
try:
    # Prevent "RuntimeError: operator torchvision::nms does not exist"
    # by mocking the module that registers these ops at import time.
    dummy_meta = types.ModuleType("torchvision._meta_registrations")
    sys.modules["torchvision._meta_registrations"] = dummy_meta
    sys.modules["torchvision.ops._register_custom_ops"] = dummy_meta
except Exception:
    pass

# Now we can import torchvision without crashing
try:
    import torchvision
    # Apply our Python NMS patch
    import nms_patch
    nms_patch.apply_patch()
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Failed to patch torchvision: {e}")

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from coordinate_calculator import offset_gps

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ─── GPU / PyTorch detection ──────────────────────────────────────────────────
try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = "cpu"

# ─── YOLO model ───────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Model path — defaults to models/best.pt (relative to this file's directory)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt"),
)

yolo_model = None

if YOLO_AVAILABLE:
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading YOLO model from: {MODEL_PATH}")
            yolo_model = YOLO(MODEL_PATH)
            yolo_model.to(DEVICE)
            logger.info(f"✓ YOLO model loaded on {DEVICE.upper()}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH} — detection disabled")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        yolo_model = None
else:
    logger.warning("ultralytics not installed — YOLO detection disabled")

# Log GPU info once
if TORCH_AVAILABLE:
    logger.info(f"PyTorch {torch.__version__}  |  CUDA {'✓ ' + torch.cuda.get_device_name(0) if CUDA_AVAILABLE else '✗ not available'}")
else:
    logger.info("PyTorch not installed — CPU only")

# ─── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Street Asset Detection Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Statistics ────────────────────────────────────────────────────────────────
stats = {
    "frames_received": 0,
    "last_frame_time": None,
    "total_bytes": 0,
    "gpu_available": CUDA_AVAILABLE,
    "processing_time_ms": [],
}


# ─── Inference endpoint ───────────────────────────────────────────────────────
@app.post("/infer")
async def infer(
    frame: UploadFile = File(...),
    session_id: str = Form("unknown"),
    frame_number: int = Form(0),
    gps_lat: float = Form(0.0),
    gps_lon: float = Form(0.0),
    gps_accuracy: float = Form(0.0),
    gps_timestamp: str = Form(""),
    bearing: float = Form(0.0),
    left_offset_m: float = Form(6.0),
    right_offset_m: float = Form(2.0),
):
    """Receive a video frame from the Flutter app and return YOLO detections."""
    try:
        # ── Read image bytes ──────────────────────────────────────────────
        frame_data = await frame.read()
        if not frame_data:
            return JSONResponse({"error": "Empty frame data"}, status_code=400)

        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Failed to decode image"}, status_code=400)

        # ── Update stats ──────────────────────────────────────────────────
        stats["frames_received"] += 1
        stats["last_frame_time"] = datetime.now().isoformat()
        stats["total_bytes"] += len(frame_data)

        # ── GPS data dict ─────────────────────────────────────────────────
        gps_data = {
            "latitude": gps_lat,
            "longitude": gps_lon,
            "accuracy": gps_accuracy,
            "timestamp": gps_timestamp or datetime.now().isoformat(),
        }

        # ── Run detection ─────────────────────────────────────────────────
        from fastapi.concurrency import run_in_threadpool
        t0 = time.time()
        detections = await run_in_threadpool(
            _process_frame,
            frame=img,
            gps_data=gps_data,
            bearing=bearing,
            left_offset_m=left_offset_m,
            right_offset_m=right_offset_m,
        )
        elapsed_ms = (time.time() - t0) * 1000

        # Keep last 100 timing samples
        stats["processing_time_ms"].append(elapsed_ms)
        if len(stats["processing_time_ms"]) > 100:
            stats["processing_time_ms"] = stats["processing_time_ms"][-100:]

        gpu_note = ""
        if CUDA_AVAILABLE:
            gpu_note = (
                f", GPU: {torch.cuda.get_device_name(0)}, "
                f"Mem: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB"
            )

        logger.info(
            f"Frame {frame_number} (session {session_id}) → "
            f"{img.shape}, {elapsed_ms:.1f}ms, "
            f"{len(detections)} det, {DEVICE.upper()}{gpu_note}"
        )

        return {"frame_number": frame_number, "detections": detections}

    except Exception as e:
        import traceback

        logger.error(f"Error processing frame: {e}\n{traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── Core detection logic ─────────────────────────────────────────────────────
def _process_frame(
    frame: np.ndarray,
    gps_data: dict,
    bearing: float = 0.0,
    left_offset_m: float = 6.0,
    right_offset_m: float = 2.0,
) -> list:
    """
    Run YOLO inference on a BGR frame and return detections with GPS coordinates.

    Returns list of dicts matching the Flutter app's expected format:
        {bbox: [x1,y1,x2,y2], object_type, confidence, object_latitude, object_longitude}
    """
    if yolo_model is None:
        return []

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_model(frame_rgb, conf=0.25, verbose=False)

        frame_h, frame_w = frame.shape[:2]
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = (
                    yolo_model.names[class_id]
                    if hasattr(yolo_model, "names")
                    else f"class_{class_id}"
                )

                # Normalised bounding box (0-1) for Flutter overlay
                bbox_norm = [
                    float(x1 / frame_w),
                    float(y1 / frame_h),
                    float(x2 / frame_w),
                    float(y2 / frame_h),
                ]

                # ── GPS offset calculation ────────────────────────────────
                bbox_cx = (x1 + x2) / 2.0
                is_left = bbox_cx < (frame_w / 2.0)
                offset_dist = left_offset_m if is_left else right_offset_m

                bearing_rad = math.radians(bearing)
                if is_left:
                    off_n = offset_dist * math.sin(bearing_rad)
                    off_e = -offset_dist * math.cos(bearing_rad)
                else:
                    off_n = -offset_dist * math.sin(bearing_rad)
                    off_e = offset_dist * math.cos(bearing_rad)

                # Validate GPS
                dev_lat = gps_data.get("latitude", 0.0)
                dev_lon = gps_data.get("longitude", 0.0)
                if dev_lat == 0.0 and dev_lon == 0.0:
                    continue
                if not (-90 <= dev_lat <= 90) or not (-180 <= dev_lon <= 180):
                    continue

                try:
                    obj_lat, obj_lon = offset_gps(dev_lat, dev_lon, off_n, off_e)
                    obj_lat = round(obj_lat, 10)
                    obj_lon = round(obj_lon, 10)
                except Exception:
                    continue

                detections.append(
                    {
                        "bbox": bbox_norm,
                        "object_type": class_name,
                        "confidence": round(confidence, 3),
                        "object_latitude": obj_lat,
                        "object_longitude": obj_lon,
                    }
                )

        return detections

    except Exception as e:
        import traceback

        logger.error(f"YOLO processing error: {e}\n{traceback.format_exc()}")
        return []


# ─── Health & stats endpoints ─────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Health check — used by the Flutter app's periodic health checker."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cuda_available": CUDA_AVAILABLE,
    }


@app.get("/api/stats")
async def get_stats():
    """Server statistics."""
    avg_ms = 0.0
    if stats["processing_time_ms"]:
        avg_ms = sum(stats["processing_time_ms"]) / len(stats["processing_time_ms"])

    gpu_info = {}
    if CUDA_AVAILABLE and TORCH_AVAILABLE:
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_mb": round(
                torch.cuda.memory_allocated(0) / 1024**2, 2
            ),
            "gpu_memory_reserved_mb": round(
                torch.cuda.memory_reserved(0) / 1024**2, 2
            ),
            "cuda_version": torch.version.cuda,
        }

    return {
        "frames_received": stats["frames_received"],
        "last_frame_time": stats["last_frame_time"],
        "total_bytes": stats["total_bytes"],
        "gpu_available": stats["gpu_available"],
        "avg_processing_time_ms": round(avg_ms, 2),
        "cuda_available": CUDA_AVAILABLE,
        "processing_device": "GPU" if CUDA_AVAILABLE else "CPU",
        "model_loaded": yolo_model is not None,
        **gpu_info,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Street Asset Detection — Jetson Nano GPU Server")
    print("=" * 55)

    if TORCH_AVAILABLE:
        print(f"  PyTorch:  {torch.__version__}")
        if CUDA_AVAILABLE:
            print(f"  GPU:      {torch.cuda.get_device_name(0)}")
            print(f"  CUDA:     {torch.version.cuda}")
        else:
            print("  GPU:      ✗  (running on CPU)")
    else:
        print("  PyTorch:  not installed (CPU only)")

    if yolo_model is not None:
        print(f"  Model:    ✓  {MODEL_PATH}")
    else:
        print(f"  Model:    ✗  not loaded")

    print("=" * 55)
    print("  Endpoints:")
    print("    POST /infer       — receive frames & return detections")
    print("    GET  /api/health  — health check")
    print("    GET  /api/stats   — server statistics")
    print("=" * 55)
    print(f"  Listening on  http://10.42.0.1:8000")
    print("=" * 55)

    uvicorn.run(app, host="10.42.0.1", port=8000, log_level="info")
