# Footfall Counter using Computer Vision

A computer vision-based system that counts the number of people entering and exiting through a specific area in a video stream.

## Features

- Real-time person detection using YOLOv8
- Object tracking to maintain identity across frames
- Configurable counting line/region
- Real-time visualization of detections and counts
- Support for both video files and webcam input
- Save processed video with annotations

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics (for YOLOv8)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/footfall-counter.git
   cd footfall-counter
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

To run the footfall counter on a video file:
```bash
python footfall_counter.py --source path/to/your/video.mp4 --output output.avi
```

To use your webcam (default):
```bash
python footfall_counter.py
```

### Command Line Arguments

- `--source`: Path to input video file or camera index (default: 0 for webcam)
- `--output`: Path to save the output video (default: output/output.avi)

### Customizing the Counting Line

By default, a horizontal line is drawn in the middle of the frame. To change this, modify the `set_counting_line` method call in the `setup_video` method of the `FootfallCounter` class.

## How It Works

1. **Detection**: YOLOv8 is used to detect people in each frame.
2. **Tracking**: A custom tracker maintains the identity of each person across frames using centroid tracking.
3. **Counting**: The system counts when a person crosses a predefined line in the frame.
4. **Visualization**: The processed video shows detections, tracks, and counts in real-time.

## Example

Here's an example of how to use the footfall counter with a sample video:

```bash
# Process a video file and save the output
python footfall_counter.py --source sample_videos/mall_entrance.mp4 --output results/mall_counting.avi

# View the processed video
# (On Windows)
start results/mall_counting.avi
```

## Performance

- On a modern CPU, the system can process approximately 10-15 FPS with YOLOv8n (nano).
- For better performance, use a CUDA-enabled GPU by installing the appropriate PyTorch version.

## Customization

- **Model**: Change the YOLO model by modifying the `model_name` parameter in the `Detector` class.
- **Tracking**: Adjust tracking parameters like `max_disappeared` and `max_distance` in the `Tracker` class.
- **Visualization**: Customize colors, text, and other visual elements in the `Visualizer` class.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [OpenCV](https://opencv.org/) for computer vision utilities
- [PyTorch](https://pytorch.org/) for deep learning framework
