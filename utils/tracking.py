import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class Tracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize the object tracker.
        
        Args:
            max_disappeared: Maximum number of frames a track can be marked as "disappeared"
                            before being removed
            max_distance: Maximum distance between centroids to consider as the same object
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store object IDs and their centroids
        self.disappeared = {}  # Dictionary to store the number of consecutive frames an object is missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def _calculate_centroid(self, bbox: List[int]) -> Tuple[float, float]:
        """Calculate the centroid of a bounding box [x, y, w, h]."""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update the tracker with new detections.
        
        Args:
            detections: List of detection dictionaries, each containing 'bbox' and 'confidence'
            
        Returns:
            List of tracked objects with their updated IDs and states
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove objects that have disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._remove_object(object_id)
            
            return []
        
        # Initialize arrays for centroids and bounding boxes
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_boxes = []
        
        # Extract centroids and bounding boxes from detections
        for i, detection in enumerate(detections):
            centroid = self._calculate_centroid(detection['bbox'])
            input_centroids[i] = centroid
            input_boxes.append(detection['bbox'])
        
        # If no existing objects, register all detections as new objects
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self._register_object(input_centroids[i], input_boxes[i], detections[i])
        else:
            # Get object IDs and their corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id]['centroid'] for obj_id in object_ids]
            
            # Calculate pairwise distances between existing objects and new detections
            distance_matrix = np.zeros((len(object_ids), len(input_centroids)))
            for i in range(len(object_ids)):
                for j in range(len(input_centroids)):
                    distance_matrix[i, j] = self._calculate_distance(
                        object_centroids[i], 
                        input_centroids[j]
                    )
            
            # Find the minimum distance for each row and column
            rows = distance_matrix.min(axis=1).argsort()
            cols = distance_matrix.argmin(axis=1)[rows]
            
            # Track which rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # Update existing objects based on minimum distance
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if distance_matrix[row, col] > self.max_distance:
                    continue
                    
                # Update the object's centroid and bounding box
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['bbox'] = input_boxes[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle objects that have disappeared
            unused_rows = set(range(distance_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(distance_matrix.shape[1])).difference(used_cols)
            
            # If we have more objects than detections, mark the extra objects as disappeared
            if distance_matrix.shape[0] >= distance_matrix.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    # Remove objects that have disappeared for too long
                    if self.disappeared[object_id] > self.max_disappeared:
                        self._remove_object(object_id)
            
            # If we have more detections than objects, register new objects
            else:
                for col in unused_cols:
                    self._register_object(input_centroids[col], input_boxes[col], detections[col])
        
        # Return the list of tracked objects with their updated states
        tracked_objects = []
        for object_id, obj in self.objects.items():
            tracked_objects.append({
                'track_id': object_id,
                'bbox': obj['bbox'],
                'centroid': obj['centroid'],
                'confidence': obj['confidence']
            })
        
        return tracked_objects
    
    def _register_object(self, centroid: Tuple[float, float], bbox: List[int], detection: Dict):
        """Register a new object to be tracked."""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'confidence': detection.get('confidence', 0.0)
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def _remove_object(self, object_id: int):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
