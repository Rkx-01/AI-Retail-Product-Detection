import cv2
import numpy as np
from ultralytics import YOLO

class ProductDetector:
    def __init__(self, model_path='yolov8n.pt'):
        import torch
        torch.set_num_threads(1)
        
        # Load the YOLOv8 model (standard Ultralytics)
        self.model = YOLO(model_path)
        
    def detect_products(self, image, conf_threshold=0.25):
        # image is already a numpy array from cv2.imread in run_pipeline
        if image is None:
            return []
            
        # Run inference
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        
        products = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Convert to integer coordinates
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = float(box.conf[0])
                
                products.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
                
        return products
