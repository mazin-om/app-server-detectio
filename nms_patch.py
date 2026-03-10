"""
Pure-Python NMS patch for torchvision on Jetson.
Replaces torchvision.ops.nms when the C++ extension is broken.
"""
import torch
import logging

logger = logging.getLogger(__name__)


def py_nms(boxes, scores, iou_threshold):
    """
    Pure PyTorch implementation of NMS to replace torchvision.ops.nms
    when the C++ extension is missing or broken.
    
    Args:
        boxes: Tensor[N, 4] in (x1, y1, x2, y2) format
        scores: Tensor[N]
        iou_threshold: float
    Returns:
        Tensor of indices to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Ensure 2D boxes
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)

    n = boxes.shape[0]
    if n == 1:
        return torch.tensor([0], dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by score descending
    _, order = scores.sort(descending=True)
    # Ensure order is always 1D
    order = order.reshape(-1)

    keep = []

    while order.numel() > 0:
        # Get the index of the highest scoring remaining box
        idx = order[0].item()
        keep.append(idx)

        if order.numel() == 1:
            break

        # Remaining indices
        rest = order[1:]

        # Compute IoU of the picked box with the rest
        xx1 = torch.clamp(x1[rest], min=x1[idx].item())
        yy1 = torch.clamp(y1[rest], min=y1[idx].item())
        xx2 = torch.clamp(x2[rest], max=x2[idx].item())
        yy2 = torch.clamp(y2[rest], max=y2[idx].item())

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        iou = inter / (areas[idx] + areas[rest] - inter + 1e-6)

        # Keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = rest[mask]
        # Ensure order stays 1D
        order = order.reshape(-1)

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def apply_patch():
    """Check if torchvision.ops.nms works, if not, patch it."""
    try:
        import torchvision
    except Exception:
        logger.warning("torchvision not available, skipping NMS patch")
        return

    try:
        # Test with dummy data on CPU
        b = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        s = torch.tensor([1.0], dtype=torch.float32)
        torchvision.ops.nms(b, s, 0.5)
        logger.info("✓ torchvision.ops.nms is working natively")
    except Exception as e:
        logger.warning(f"⚠ torchvision.ops.nms failed ({e}). Monkey-patching with pure Python NMS.")
        torchvision.ops.nms = py_nms

        # Verify patch works
        try:
            b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], dtype=torch.float32)
            s = torch.tensor([0.9, 0.8], dtype=torch.float32)
            result = py_nms(b, s, 0.5)
            logger.info(f"✓ Monkey-patch successful (test result: {result})")
        except Exception as e2:
            logger.error(f"❌ Monkey-patch verification failed: {e2}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_patch()
