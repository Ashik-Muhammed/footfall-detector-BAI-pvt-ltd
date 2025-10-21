import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class Visualizer:
    def __init__(self):
        """Initialize the visualizer with default colors and settings."""
        # Color scheme
        self.COLORS = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        
        # Visualization settings
        self.BOX_THICKNESS = 2
        self.TEXT_THICKNESS = 1
        self.TEXT_SCALE = 0.6
        self.LINE_THICKNESS = 2
        self.DOT_RADIUS = 4
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection bounding boxes on the frame.
        
        Args:
            frame: Input BGR image
            detections: List of detections, each with 'bbox' and 'confidence'
            
        Returns:
            Frame with detection boxes drawn
        """
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection.get('confidence', 0)
            
            # Draw bounding box
            cv2.rectangle(
                frame, 
                (x, y), (x + w, y + h), 
                self.COLORS['green'], 
                self.BOX_THICKNESS
            )
            
            # Draw confidence score
            label = f"{confidence:.2f}"
            self._draw_text_with_background(
                frame, 
                label, 
                (x, y - 5), 
                self.TEXT_SCALE, 
                self.COLORS['green']
            )
        
        return frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """
        Draw tracking information on the frame.
        
        Args:
            frame: Input BGR image
            tracks: List of tracked objects with 'track_id' and 'centroid'
            
        Returns:
            Frame with tracking information drawn
        """
        for track in tracks:
            track_id = track['track_id']
            x, y, w, h = track['bbox']
            centroid = track.get('centroid', (x + w//2, y + h//2))
            
            # Draw tracking ID
            self._draw_text_with_background(
                frame, 
                f"ID: {track_id}", 
                (x, y - 25 if y > 30 else y + h + 5), 
                self.TEXT_SCALE, 
                self.COLORS['yellow']
            )
            
            # Draw centroid
            cv2.circle(
                frame, 
                (int(centroid[0]), int(centroid[1])), 
                self.DOT_RADIUS, 
                self.COLORS['red'], 
                -1
            )
            
            # Draw direction indicator if available
            if 'direction' in track:
                direction = track['direction']
                end_point = (
                    int(centroid[0] + 30 * np.cos(direction)),
                    int(centroid[1] + 30 * np.sin(direction))
                )
                cv2.arrowedLine(
                    frame, 
                    (int(centroid[0]), int(centroid[1])), 
                    end_point, 
                    self.COLORS['blue'], 
                    2, 
                    tipLength=0.3
                )
        
        return frame
    
    def draw_counting_line(self, frame: np.ndarray, line: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Draw the counting line on the frame.
        
        Args:
            frame: Input BGR image
            line: Tuple of (x1, y1, x2, y2) coordinates for the counting line
            
        Returns:
            Frame with counting line drawn
        """
        x1, y1, x2, y2 = line
        
        # Draw the line
        cv2.line(
            frame, 
            (x1, y1), (x2, y2), 
            self.COLORS['red'], 
            self.LINE_THICKNESS
        )
        
        # Add label
        self._draw_text_with_background(
            frame, 
            "Counting Line", 
            (x1, y1 - 10), 
            self.TEXT_SCALE, 
            self.COLORS['red']
        )
        
        return frame
    
    def draw_counts(self, frame: np.ndarray, entries: int, exits: int) -> np.ndarray:
        """
        Draw the entry and exit counts on the frame.
        
        Args:
            frame: Input BGR image
            entries: Number of entries
            exits: Number of exits
            
        Returns:
            Frame with counts drawn
        """
        # Create a semi-transparent background for the counts
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (10, 10), (250, 100), 
            self.COLORS['black'], 
            -1
        )
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw counts
        self._draw_text(
            frame, 
            f"Entries: {entries}", 
            (20, 40), 
            self.TEXT_SCALE * 1.2, 
            self.COLORS['green']
        )
        
        self._draw_text(
            frame, 
            f"Exits: {exits}", 
            (20, 70), 
            self.TEXT_SCALE * 1.2, 
            self.COLORS['red']
        )
        
        self._draw_text(
            frame, 
            f"Total: {entries - exits}", 
            (20, 100), 
            self.TEXT_SCALE * 1.2, 
            self.COLORS['white']
        )
        
        return frame
    
    def _draw_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                  scale: float, color: Tuple[int, int, int], 
                  thickness: Optional[int] = None) -> None:
        """Helper function to draw text on the frame."""
        if thickness is None:
            thickness = self.TEXT_THICKNESS
            
        cv2.putText(
            frame, 
            text, 
            position, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            scale, 
            color, 
            thickness, 
            cv2.LINE_AA
        )
    
    def _draw_text_with_background(self, frame: np.ndarray, text: str, 
                                 position: Tuple[int, int], scale: float, 
                                 color: Tuple[int, int, int], 
                                 bg_color: Optional[Tuple[int, int, int]] = None) -> None:
        """Helper function to draw text with a background rectangle."""
        if bg_color is None:
            bg_color = self.COLORS['black']
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            text, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            scale, 
            self.TEXT_THICKNESS
        )
        
        # Calculate background rectangle coordinates
        x, y = position
        padding = 2
        cv2.rectangle(
            frame, 
            (x - padding, y - text_height - padding), 
            (x + text_width + padding, y + padding), 
            bg_color, 
            -1
        )
        
        # Draw text
        self._draw_text(
            frame, 
            text, 
            (x, y), 
            scale, 
            color, 
            self.TEXT_THICKNESS
        )
