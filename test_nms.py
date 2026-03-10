
import torch
import torchvision
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test():
    print(f"Torch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    # Create dummy boxes and scores
    boxes = torch.tensor([[0, 0, 100, 100], [10, 10, 110, 110]], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float)
    
    if torch.cuda.is_available():
        boxes = boxes.cuda()
        scores = scores.cuda()
        print("Moved tensors to CUDA")
    
    try:
        # Run NMS
        keep = torchvision.ops.nms(boxes, scores, 0.5)
        print(f"NMS success! Keep: {keep}")
        return True
    except Exception as e:
        print(f"NMS failed: {e}")
        return False

if __name__ == "__main__":
    test()
