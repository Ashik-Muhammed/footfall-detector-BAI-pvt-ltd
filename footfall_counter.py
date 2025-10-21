import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from utils.detection import Detector
from utils.tracking import Tracker
from utils.visualization import Visualizer

class FootfallCounter:
    def __init__(self, source: str = None, output_path: str = "output/output.avi"):
        # Default video path
        default_video = r"C:\Users\Ashik\Desktop\background video _ people _ walking _.mp4"
        
        # If source is not provided, use the default video
        if source is None:
            import os
            if os.path.exists(default_video):
                source = default_video
                print(f"Using default video file: {source}")
            else:
                print(f"Error: Default video file not found at {default_video}")
                print("Please provide a valid video file using --source argument.")
                exit(1)
        """
        Initialize the Footfall Counter.
        
        Args:
            source: Path to video file or camera index (0 for webcam)
            output_path: Path to save the output video
        """
        self.source = source
        self.output_path = output_path
        self.detector = Detector()
        self.tracker = Tracker()
        self.visualizer = Visualizer()
        
        # Video properties
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.frame_count = 0
        
        # Counting line/region
        self.counting_line = None
        self.entries = 0
        self.exits = 0
        
    def setup_video(self):
        """Initialize video capture and writer."""
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Attempting to open video source: {self.source}")
        
        # Try to determine if source is a file or camera index
        try:
            # If source can be converted to int, it's a camera index
            source = int(self.source)
            print(f"Source is a camera index: {source}")
        except ValueError:
            # Otherwise, it's a file path
            source = str(self.source)
            if not Path(source).exists():
                raise FileNotFoundError(f"Video file not found: {source}")
            print(f"Source is a video file: {source}")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video source {source}")
            
        print(f"Successfully opened video source. Properties:")
        print(f"- Frame width: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
        print(f"- Frame height: {int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"- FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"- Total frames: {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print(f"- Format: {self.cap.get(cv2.CAP_PROP_FORMAT)}")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Default counting line (horizontal, middle of the frame)
        self.counting_line = (0, self.frame_height // 2, self.frame_width, self.frame_height // 2)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            (self.frame_width, self.frame_height)
        )
    
    def set_counting_line(self, line: Tuple[int, int, int, int]):
        """Set the counting line coordinates (x1, y1, x2, y2)."""
        self.counting_line = line
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return the annotated frame."""
        if frame is None:
            print("Warning: Received empty frame")
            return None
            
        # Make a copy of the frame for display
        display_frame = frame.copy()
        
        # Detect people in the frame
        detections = self.detector.detect(frame)
        print(f"Detected {len(detections)} people in the frame")
        
        # Update tracker with new detections
        tracks = self.tracker.update(detections)
        print(f"Tracking {len(tracks)} people")
        
        # Update counters based on track movements
        self._update_counts(tracks)
        
        # Draw visualizations
        if hasattr(self, 'visualizer'):
            display_frame = self.visualizer.draw_detections(display_frame, detections)
            display_frame = self.visualizer.draw_tracks(display_frame, tracks)
            display_frame = self.visualizer.draw_counting_line(display_frame, self.counting_line)
            display_frame = self.visualizer.draw_counts(display_frame, self.entries, self.exits)
        else:
            print("Warning: Visualizer not initialized")
            # Draw basic info if visualizer is not available
            cv2.putText(display_frame, f'People: {len(detections)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Entries: {self.entries}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Exits: {self.exits}', (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display_frame
    
    def _update_counts(self, tracks: List[Dict]):
        """Update entry and exit counts based on track movements across the counting line."""
        if not hasattr(self, 'track_history'):
            self.track_history = {}
            
        x1, y1, x2, y2 = self.counting_line
        
        for track in tracks:
            track_id = track['track_id']
            x_center = track['bbox'][0] + track['bbox'][2] // 2
            y_center = track['bbox'][1] + track['bbox'][3] // 2
            
            # Initialize track history if not exists
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'positions': [],
                    'counted': False
                }
            
            # Add current position to track history
            self.track_history[track_id]['positions'].append((x_center, y_center))
            
            # Only keep the last 10 positions to track movement direction
            if len(self.track_history[track_id]['positions']) > 10:
                self.track_history[track_id]['positions'].pop(0)
            
            # Only count if not already counted
            if not self.track_history[track_id]['counted']:
                # Get the first and last positions
                positions = self.track_history[track_id]['positions']
                if len(positions) >= 2:
                    first_y = positions[0][1]
                    last_y = positions[-1][1]
                    
                    # If person crossed the line from top to bottom (entry)
                    if first_y < y1 and last_y >= y1:
                        self.entries += 1
                        self.track_history[track_id]['counted'] = True
                        print(f"Person {track_id} entered! Total entries: {self.entries}")
                    # If person crossed the line from bottom to top (exit)
                    elif first_y > y1 and last_y <= y1:
                        self.exits += 1
                        self.track_history[track_id]['counted'] = True
                        print(f"Person {track_id} exited! Total exits: {self.exits}")
            
            # Check if track has crossed the counting line
            if 'prev_position' in track:
                prev_y = track['prev_position'][1]
                
                # Check if track crossed the line from top to bottom (entry)
                if prev_y <= y1 and y_center > y1:
                    self.entries += 1
                # Check if track crossed the line from bottom to top (exit)
                elif prev_y >= y1 and y_center < y1:
                    self.exits += 1
            
            # Update previous position for next frame
            track['prev_position'] = (x_center, y_center)
    
    def run(self):
        """Run the footfall counter on the video source."""
        try:
            self.setup_video()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write frame to output video
                self.out.write(processed_frame)
                
                # Display the frame
                cv2.imshow('Footfall Counter', processed_frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            print(f"\nProcessing complete!")
            print(f"Total entries: {self.entries}")
            print(f"Total exits: {self.exits}")
            print(f"Output video saved to: {self.output_path}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
        finally:
            # Release resources
            if self.cap is not None:
                self.cap.release()
            if hasattr(self, 'out') and self.out is not None:
                self.out.release()
            cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Footfall Counter using Computer Vision')
    parser.add_argument('--source', type=str, default=None, 
                        help='Path to video file or camera index (default: looks for video files in current directory)')
    parser.add_argument('--output', type=str, default='output/output.avi',
                        help='Path to save the output video (default: output/output.avi)')
    
    args = parser.parse_args()
    
    # Initialize and run the footfall counter
    counter = FootfallCounter(source=args.source, output_path=args.output)
    counter.run()


if __name__ == "__main__":
    main()
