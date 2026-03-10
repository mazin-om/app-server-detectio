#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Street Asset Detection — Jetson Nano GPU Server Launcher
# ═══════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Model symlink setup ──────────────────────────────────────
MODEL_DIR="$SCRIPT_DIR/models"
MODEL_FILE="$MODEL_DIR/best.pt"
SOURCE_MODEL="$SCRIPT_DIR/../YOLOv8/F12YOLO/runs/detect/street_assets/weights/best.pt"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ] && [ -f "$SOURCE_MODEL" ]; then
    echo "→ Linking model: $SOURCE_MODEL"
    ln -sf "$(realpath "$SOURCE_MODEL")" "$MODEL_FILE"
elif [ ! -f "$MODEL_FILE" ]; then
    echo "⚠  Model not found at $SOURCE_MODEL"
    echo "   Place your best.pt model in $MODEL_DIR/"
fi

# ── Activate virtualenv if present ────────────────────────────
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "→ Using virtualenv: $SCRIPT_DIR/venv"
elif [ -d "$HOME/venv" ]; then
    source "$HOME/venv/bin/activate"
    echo "→ Using virtualenv: $HOME/venv"
fi



# ── Launch server ─────────────────────────────────────────────
echo ""
echo "Starting inference server on http://0.0.0.0:8000 ..."
echo ""

exec python3 server.py
