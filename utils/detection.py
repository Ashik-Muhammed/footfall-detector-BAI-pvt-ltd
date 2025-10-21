import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

class Detector:
    def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        """
        Initialize the person detector using YOLOv8.
        
        Args:
            model_name: Name of the YOLOv8 model to use (default: 'yolov8n.pt')
            conf_threshold: Confidence threshold for detections (default: 0.5)
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.class_id = 0  # 0 is the class ID for 'person' in COCO dataset
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in the given frame.
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of detections, where each detection is a dictionary with:
            - 'bbox': [x, y, width, height] in pixel coordinates
            - 'confidence': Detection confidence score
            - 'class_id': Class ID (0 for person)
        """
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = result
            
            # Filter out non-person detections and low-confidence detections
            if int(class_id) == self.class_id and conf >= self.conf_threshold:
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # [x, y, w, h]
                    'confidence': float(conf),
                    'class_id': int(class_id)
                })
        
        return detections
