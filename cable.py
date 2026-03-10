#!/usr/bin/env python3
"""
cable.py — Direct Android camera detection via IP Webcam
=========================================================
Use this to test YOLO detection directly from your phone camera,
bypassing the Flutter app entirely.  This helps isolate whether
accuracy issues come from the app or the AI model.

Setup:
  1. Install "IP Webcam" app on your Android phone (from Play Store)
  2. Open the app and tap "Start server" at the bottom
  3. Note the URL shown (e.g. http://192.168.x.x:8080)
  4. Connect phone to Jetson via USB cable and enable USB tethering,
     OR connect both to the same WiFi / Jetson hotspot
  5. Run:  python3 cable.py <URL>
     e.g:  python3 cable.py http://192.168.1.100:8080

Controls:
  q / ESC  — quit
  s        — save current frame + detections to disk
  +/-      — increase/decrease confidence threshold
"""

import sys
import os
import time
import types
import argparse

# ─── Torchvision compatibility patch (same as server.py) ───
try:
    dummy = types.ModuleType("torchvision._meta_registrations")
    sys.modules["torchvision._meta_registrations"] = dummy
    sys.modules["torchvision.ops._register_custom_ops"] = dummy
except Exception:
    pass

import cv2
import numpy as np

# Patch NMS before importing ultralytics
try:
    import nms_patch
    nms_patch.apply_patch()
except Exception:
    pass

from ultralytics import YOLO

# ─── Class names ────────────────────────────────────────────
CLASSES_FILE = os.path.join(os.path.dirname(__file__),
                            "..", "YOLOv8", "F12YOLO", "classes.txt")

def load_classes(path):
    if os.path.exists(path):
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]
    return []

CLASS_NAMES = load_classes(CLASSES_FILE)

# ─── Color palette (one per class) ─────────────────────────
np.random.seed(42)
COLORS = [(int(r), int(g), int(b))
          for r, g, b in np.random.randint(60, 255, size=(len(CLASS_NAMES) + 50, 3))]


def draw_detections(frame, results, conf_thresh):
    """Draw bounding boxes, labels, and confidence on frame."""
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            color = COLORS[cls_id % len(COLORS)]
            label = f"{name} {conf:.0%}"

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

            detections.append((name, conf, (x1, y1, x2, y2)))

    return detections


def draw_hud(frame, fps, conf_thresh, n_det, device_str):
    """Draw heads-up display overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    text = (f"FPS: {fps:.1f}  |  Conf: {conf_thresh:.0%}  |  "
            f"Det: {n_det}  |  {device_str}  |  [q]uit [s]ave [+/-] conf")
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="Direct phone camera YOLO detection (bypass Flutter app)")
    parser.add_argument("url", nargs="?",
                        default="http://192.168.1.100:8080",
                        help="IP Webcam URL (e.g. http://192.168.1.100:8080)")
    parser.add_argument("--model", default=None,
                        help="Path to YOLO model (default: server/models/best.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Initial confidence threshold (default: 0.25)")
    parser.add_argument("--size", type=int, default=640,
                        help="Inference size (default: 640)")
    args = parser.parse_args()

    # ── Model ─────────────────────────────────────────────────
    model_path = args.model or os.path.join(os.path.dirname(__file__),
                                            "models", "best.pt")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        device_str = f"GPU: {gpu_name}"
    else:
        device_str = "CPU"
    print(f"Device: {device_str}")

    # ── Camera stream ─────────────────────────────────────────
    # IP Webcam serves MJPEG at /video
    stream_url = args.url.rstrip("/") + "/video"
    print(f"\nConnecting to: {stream_url}")
    print("(Make sure IP Webcam is running on your phone)\n")

    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

    if not cap.isOpened():
        # Try the raw URL
        print(f"Could not open {stream_url}, trying {args.url}...")
        cap = cv2.VideoCapture(args.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Cannot connect to camera at {args.url}")
        sys.exit(1)

    print("✓ Connected to camera!")
    print("  Press 'q' to quit, 's' to save frame, '+'/'-' to adjust confidence\n")

    conf_thresh = args.conf
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_frames = 0
    save_dir = os.path.join(os.path.dirname(__file__), "cable_captures")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Lost connection, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            frame_count += 1

            # ── Inference ─────────────────────────────────────
            t0 = time.time()
            results = model(frame, conf=conf_thresh, imgsz=args.size,
                            verbose=False, device=device)
            dt = time.time() - t0

            # ── Draw ─────────────────────────────────────────
            detections = draw_detections(frame, results, conf_thresh)
            draw_hud(frame, fps, conf_thresh, len(detections), device_str)

            # ── FPS calculation ───────────────────────────────
            fps_frames += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_timer = time.time()

            # ── Print detections to terminal ──────────────────
            if detections:
                det_str = ", ".join(f"{n}({c:.0%})" for n, c, _ in detections)
                print(f"[F{frame_count:04d}] {dt*1000:.0f}ms | {det_str}")

            # ── Display ──────────────────────────────────────
            cv2.imshow("Cable Detection - YOLO Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('s'):
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"capture_{frame_count:04d}.jpg")
                cv2.imwrite(fname, frame)
                print(f"  💾 Saved: {fname}")
            elif key == ord('+') or key == ord('='):
                conf_thresh = min(0.95, conf_thresh + 0.05)
                print(f"  Confidence: {conf_thresh:.0%}")
            elif key == ord('-'):
                conf_thresh = max(0.05, conf_thresh - 0.05)
                print(f"  Confidence: {conf_thresh:.0%}")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()
