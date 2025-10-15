# Weapon Detection System

A real-time weapon detection system using OpenCV and Haar Cascade classifier. This project uses your webcam to detect weapons and displays visual alerts on the video feed.

## Features

- Real-time weapon detection using webcam
- Visual alerts with bounding boxes around detected objects
- On-screen status display ("GUN DETECTED!" or "NO GUN")
- Optimized detection parameters to minimize false positives
- Comprehensive beginner-friendly comments explaining every function

## Requirements

- Python 3.12 or higher
- OpenCV (cv2)
- NumPy
- Imutils

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd "AI assest"
```

2. Install required packages:
```bash
pip install opencv-python numpy imutils
```

## Usage

Run the weapon detection script:
```bash
python weapon_dection.py
```

- The webcam will open and start detecting weapons in real-time
- Press **'q'** to quit the application
- A summary will be printed showing if any weapons were detected during the session

## Files

- `weapon_dection.py` - Main weapon detection script with detailed comments
- `cascade.xml` - Haar cascade classifier for weapon detection
- `cascade_1.2.xml` - Alternative cascade classifier
- `probe_camera.py` - Camera diagnostic tool (if needed)

## How It Works

1. **Load Cascade**: Loads the pre-trained Haar cascade classifier from `cascade.xml`
2. **Open Camera**: Accesses the default webcam (index 0)
3. **Process Frames**: Continuously captures and analyzes video frames
4. **Detect Weapons**: Uses `detectMultiScale()` with optimized parameters:
   - `scaleFactor=1.05` - Image pyramid scale
   - `minNeighbors=25` - Strict detection threshold to reduce false positives
   - `minSize=(120, 120)` - Minimum object size in pixels
5. **Visual Feedback**: Draws rectangles around detected weapons and displays status text
6. **User Control**: Press 'q' to quit and see detection summary

## Detection Parameters

The detection accuracy can be adjusted by modifying these parameters in `weapon_dection.py`:

- **scaleFactor**: Lower values (e.g., 1.03) = more thorough but slower
- **minNeighbors**: Higher values = fewer false positives but might miss objects
- **minSize**: Larger values = ignore smaller objects

## Troubleshooting

**Camera not opening:**
- Ensure no other application is using the webcam
- Try different camera indices (0, 1, 2) in `cv2.VideoCapture()`
- Check camera permissions in Windows settings

**Too many false positives:**
- Increase `minNeighbors` parameter (e.g., 30 or 35)
- Increase `minSize` parameter (e.g., (150, 150))
- Ensure good lighting conditions

**Cascade file not found:**
- Verify `cascade.xml` is in the same directory as the script
- Check the file path in the code

## Educational Purpose

This project includes extensive comments explaining:
- OpenCV functions and parameters
- Image processing concepts (grayscale conversion, resizing)
- Object detection methodology
- Real-time video processing loops
- Proper resource management (camera release, window cleanup)

Perfect for beginners learning Python, OpenCV, and computer vision!

## Notes

- This is an educational project demonstrating Haar cascade object detection
- Detection accuracy depends on the quality of the cascade classifier
- Current cascade has 10 stages with 50% false alarm rate (as per XML configuration)
- For production use, consider modern deep learning approaches (YOLO, SSD, etc.)

## License

This project is for educational purposes.
