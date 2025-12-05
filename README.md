# Advanced Hand Tracking System

A real-time computer vision application that uses AI-powered hand tracking for gesture recognition, virtual drawing, volume control, and interactive gaming.


## Features

### Mode 1: Gesture Recognition & Tracking
- **Real-time gesture detection** with 10+ recognizable gestures
- **Color-coded visualization** for easy gesture identification
- **Multi-hand tracking** (up to 2 hands simultaneously)
- **Hand orientation detection** (Left/Right hand classification)
- **21-point landmark tracking** per hand
- **Live FPS monitoring** for performance tracking

**Supported Gestures:**
- ✊ Fist
- 👍 Thumbs Up / 👎 Thumbs Down
- ☝️ Pointing
- ✌️ Peace / Victory Sign
- 🔢 Number counting (1-5)
- 👌 OK Sign
- 🤘 Rock On
- 🤙 Call Me (Shaka)
- 🕷️ Spiderman

### Mode 2: Virtual Air Painter
- **Draw in the air** using your index finger
- **5 color palette**: Red, Green, Blue, Yellow, White
- **Real-time canvas overlay** on video feed
- **Smooth line drawing** with motion tracking
- **Clear canvas functionality**
- **Touch-free color selection**

### Mode 3: Volume Controller
- **Control system volume** using hand gestures
- **Pinch gesture** (thumb + index finger distance)
- **Visual feedback** with live volume bar
- **Smooth volume transitions**
- **Distance-based control** (30-300 pixels range)
- **Real-time percentage display**

### Mode 4: Interactive Game
- **Target shooting game** controlled by hand gestures
- **Point and shoot** with index finger
- **Dynamic target spawning**
- **Score tracking system**
- **Collision detection**
- **Increasing difficulty**


## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows OS (for volume control feature)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/advanced-hand-tracking.git
cd advanced-hand-tracking
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Core dependencies (required)
pip install opencv-python mediapipe numpy

# Volume control dependencies (optional - Windows only)
pip install pycaw comtypes
```

### Alternative: Install from requirements.txt
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
```bash
python hand_tracking.py
```

### Quick Start Guide
1. **Launch** the application
2. **Position yourself** in front of the webcam (arm's length distance recommended)
3. **Ensure good lighting** for optimal hand detection
4. **Press number keys** to switch between modes
5. **Make gestures** and see real-time recognition!

## Controls

| Key | Action | Description |
|-----|--------|-------------|
| **1** | Tracking Mode | Gesture recognition and hand analytics |
| **2** | Drawing Mode | Virtual air painter |
| **3** | Volume Control | System volume adjustment |
| **4** | Game Mode | Interactive target shooting |
| **C** | Clear/Reset | Clear canvas or reset game score |
| **Q** | Quit | Exit application |

## How It Works

### Mode-Specific Instructions

#### Tracking Mode
- Simply make gestures in front of the camera
- Watch real-time recognition with color-coded labels
- Observe finger states and hand orientation
- Multiple gestures detected simultaneously

#### Drawing Mode
1. **Extend index finger only** to draw
2. **Touch color boxes** at top to change color
3. **Make peace sign** (index + middle) to pause drawing
4. **Press C** to clear the canvas
5. Lines appear with transparency overlay

#### Volume Control Mode
1. **Extend thumb and index finger** (pinch gesture)
2. **Move fingers apart** to increase volume
3. **Bring fingers together** to decrease volume
4. **Visual bar** shows current volume level
5. Works with Windows system audio

#### Game Mode
1. **Point with index finger** at red circular targets
2. **Touch targets** to destroy them and score points
3. **Targets spawn randomly** on screen
4. **Track your score** in top-right corner
5. **Press C** to reset score

## Project Structure

```
advanced-hand-tracking/
│
├── hand_tracking.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

## Technical Details

### Technologies Used
- **OpenCV**: Real-time computer vision and image processing
- **MediaPipe**: Google's ML solution for hand landmark detection
- **NumPy**: Numerical computing and array operations
- **PyCaw**: Windows Core Audio API wrapper (volume control)

### System Architecture

```
┌─────────────────┐
│  Video Capture  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MediaPipe Hand │
│    Detection    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Landmark      │
│   Extraction    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Gesture      │
│  Recognition    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Mode-Specific  │
│    Processing   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │
│   & Display     │
└─────────────────┘
```

### Hand Landmark Model
MediaPipe detects **21 landmarks** per hand:

```
       8   12  16  20
       |   |   |   |
   4---5---9--13--17  (finger tips)
   |   |   |   |   |
   3   6--10--14--18  (joints)
   |   |   |   |   |
   2   7--11--15--19
   |
   1
   |
   0 (wrist)
```

### Gesture Recognition Algorithm
1. **Finger State Detection**: Determines which fingers are extended
2. **Pattern Matching**: Compares finger states to known patterns
3. **Distance Calculation**: Measures spacing between landmarks
4. **Orientation Analysis**: Determines hand position and direction
5. **Stability Tracking**: Confirms gesture held over multiple frames

## Performance

### Benchmarks
- **FPS**: 25-35 on average hardware (1280x720)
- **Latency**: <50ms gesture recognition
- **Accuracy**: ~95% in good lighting conditions
- **CPU Usage**: 15-25% on modern processors

### Optimization Tips
- Lower camera resolution for better FPS
- Increase detection confidence for fewer false positives
- Close background applications
- Ensure good lighting (reduces processing time)

## Troubleshooting

### Common Issues

#### Camera Not Opening
```python
# Error: Can't open camera
Solution: Check if camera is being used by another application
         Try changing camera index: VideoCapture(1) instead of VideoCapture(0)
```

#### Low FPS / Lag
```python
# Solution 1: Reduce resolution
cap.set(3, 640)   # Width
cap.set(4, 480)   # Height

# Solution 2: Increase confidence thresholds
detector = HandDetector(detection_con=0.8, track_con=0.8)
```

#### Hands Not Detected
- Ensure good lighting (avoid backlighting)
- Keep hands clearly visible (not too close or far)
- Check camera focus
- Increase hand contrast with background
- Lower detection confidence: `detection_con=0.5`

#### Volume Control Not Working
```bash
# Install Windows audio dependencies
pip install pycaw comtypes

# Run as administrator if permission issues occur
```

#### Gesture Flickering
- Increase `gesture_hold_frames` threshold
- Improve lighting conditions
- Reduce camera noise
- Increase tracking confidence

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed | `pip install opencv-python` |
| `No module named 'mediapipe'` | MediaPipe not installed | `pip install mediapipe` |
| `Camera index out of range` | Wrong camera index | Try different index (0, 1, 2) |
| `Volume control unavailable` | PyCaw not installed | `pip install pycaw comtypes` |


## 📈 Future Enhancements

### Planned Features
- [ ] Custom gesture training module
- [ ] Mouse cursor control with hands
- [ ] Sign language recognition
- [ ] Multi-hand gesture combinations
- [ ] Gesture macro recording
- [ ] Web-based interface


