# Street Asset Detection — Jetson Nano GPU Server

Inference server that receives camera frames from the Flutter app and returns YOLO detection results, processed on Jetson Nano GPU.

## Prerequisites

- Python 3.8+
- PyTorch with CUDA (from [NVIDIA Jetson wheels](https://forums.developer.nvidia.com/t/pytorch-for-jetson/))
- The trained `best.pt` model (auto-linked from `YOLOv8/F12YOLO/runs/detect/street_assets/weights/`)

## Quick Start

```bash
# 1. Install dependencies
# NOTE: torchvision might need compilation from source if pre-built wheels fail.
pip install -r requirements.txt

# 2. Start the server (auto-links model & launches on port 8000)
chmod +x start.sh
./start.sh
```

Or manually:

```bash
# Link the model
mkdir -p models
ln -sf "$(realpath ../YOLOv8/F12YOLO/runs/detect/street_assets/weights/best.pt)" models/best.pt

# Run
python3 server.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/infer` | POST | Send frame + GPS data → get detections |
| `/api/health` | GET | Health check |
| `/api/stats` | GET | Server statistics |

### POST `/infer`

**Request** — multipart/form-data:

| Field | Type | Description |
|---|---|---|
| `frame` | file | JPEG image |
| `session_id` | string | Session identifier |
| `frame_number` | int | Frame counter |
| `gps_lat` | float | Device latitude |
| `gps_lon` | float | Device longitude |
| `gps_accuracy` | float | GPS accuracy (meters) |
| `gps_timestamp` | string | ISO 8601 timestamp |
| `bearing` | float | Vehicle heading (degrees) |
| `left_offset_m` | float | Left-side object offset (meters) |
| `right_offset_m` | float | Right-side object offset (meters) |

**Response** — JSON:

```json
{
  "frame_number": 1,
  "detections": [
    {
      "bbox": [0.1, 0.2, 0.5, 0.6],
      "object_type": "stop",
      "confidence": 0.87,
      "object_latitude": 25.123456,
      "object_longitude": 55.654321
    }
  ]
}
```

## App Configuration

In the Flutter app, go to **Settings** and set the backend URL to:

```
http://10.42.0.1:8000
```

This uses the Jetson's WiFi hotspot ("jetson") IP. Connect your phone to the "jetson" WiFi hotspot first.

## Troubleshooting

| Problem | Solution |
|---|---|
| `No module named 'torch'` | Install PyTorch from NVIDIA Jetson wheels |
| `Model not found` | Run `./start.sh` to auto-link, or manually copy `best.pt` to `models/` |
| App can't connect | Ensure phone and Jetson are on the same network, firewall allows port 8000 |
| Slow inference | Verify CUDA is available: check `/api/stats` → `processing_device` should be `GPU` |
