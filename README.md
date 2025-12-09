# Advanced Hand Tracking & Game Controller

A comprehensive real-time computer vision system that combines AI-powered utilities with specialized game controllers. This project functions as an all-in-one suite for gesture recognition, virtual arts, system control, and hands-free gaming (specifically optimized for Hill Climb Racing).

## Features Overview

### Utility Modules
- **Gesture Recognition:** Real-time detection of 10+ specific hand signs with analytics
- **Virtual Air Painter:** Draw on screen using your index finger with a color palette
- **Volume Controller:** Adjust system audio by pinching fingers in the air
- **Mouse/Keyboard Simulation:** Map gestures to physical inputs

### Gaming Modules
- **Hill Climb Racing Controller:** Play driving games using hand gestures (Open Palm for Gas, Fist for Brake)
- **Target Shooting:** An interactive point-and-shoot game built directly into the CV interface
- **FPS Monitoring:** Live performance tracking for low-latency gaming

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (Internal or External)
- Windows OS (Required for Volume Control and Game Automation features)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Abdull-a-h/Hand-Tracking-System.git
cd hand-tracking-suite
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
This project requires several libraries for vision (OpenCV/MediaPipe) and input automation (PyAutoGUI/PyCaw).

```bash
pip install opencv-python mediapipe numpy pyautogui pynput pycaw comtypes
```

## Quick Start & Usage

This suite is divided into two main execution scripts based on your needs.

### Option A: The Utility Suite (Painting, Volume, Analytics)
Run the main tracking application:

```bash
python hand_tracking.py
```

**Available Modes:**
- Key `1`: Tracking & Analytics Mode
- Key `2`: Virtual Air Painter Mode
- Key `3`: Volume Control Mode
- Key `4`: Target Practice Game

### Option B: Hill Climb Racing Controller
Run the dedicated game controller script:

```bash
python hill_climbing.py
```

Follow the on-screen prompts to map the Gas and Brake pedals to your screen coordinates.

## Module Details

### Mode 1: Gesture Recognition (Analytics)
**Usage:** Simply move your hands in front of the camera.

**Features:** Displays active landmarks (21 points), hand orientation (Left/Right), and classifies gestures (Peace, Thumbs Up, OK, Spiderman, etc.).

### Mode 2: Virtual Air Painter
- **Draw:** Extend your index finger to draw
- **Select Color:** Point your finger at the color boxes (Red, Green, Blue, Yellow) at the top of the screen
- **Pause:** Show a Peace Sign (Index + Middle) to move without drawing
- **Clear:** Press `C` on the keyboard or select the Eraser tool

### Mode 3: Volume Controller
- **Activate:** Extend Thumb and Index finger
- **Adjust:** Pinch fingers (bring them closer) to lower volume; separate them to raise volume
- **Feedback:** A visual bar and percentage indicator show real-time levels

### Mode 4: Hill Climb Racing Controller
- **Setup:** On launch, select Option 1 (Mouse Control). Hover your mouse over the game's Gas Pedal and press Enter, then repeat for the Brake Pedal
- **GAS (Accelerate):** Show an Open Palm (4 or 5 fingers)
- **BRAKE (Stop):** Show a Closed Fist (0 fingers)
- **NEUTRAL (Coast):** Show partial hand (1-3 fingers)
- **Toggle:** Press `SPACE` to pause/resume control capabilities

## Controls Summary

| Key | Context | Action |
|-----|---------|--------|
| 1-4 | Utility Suite | Switch between Analytics, Paint, Volume, and Target Game |
| Space | Game Controller | Toggle game control ON/OFF (Pause) |
| R | Game Controller | Reconfigure pedal positions |
| C | Utility Suite | Clear Canvas / Reset Score |
| Q | Global | Quit Application |

## Technical Architecture

This system relies on MediaPipe for hand landmark detection. It utilizes a 21-point skeleton model to calculate gesture logic.

### Logic Flow
1. **Image Capture:** OpenCV reads frames from the webcam
2. **Detection:** MediaPipe extracts (x, y, z) coordinates for 21 hand landmarks
3. **Finger Analysis:** Algorithm determines which fingers are extended based on landmark tip vs. knuckle positions
4. **Gesture Matching:**
   - If Thumb+Index extended & distance < threshold → Volume Mode
   - If Index extended only → Drawing Mode
   - If All fingers folded → Brake (Game Mode)
5. **Action Dispatch:** Triggers PyAutoGUI (Mouse clicks) or PyCaw (System Audio)

## Project Structure

```
hand-tracking-suite/
│
├── hand_tracking.py           # Main Utility App (Modes 1-4)
├── hill_climbing.py         # Dedicated Hill Climb Racing App
├── facial_feature.py    
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Troubleshooting

### Common Issues

#### 1. "No Hand Detected"
- **Lighting:** Ensure the room is well-lit. Backlighting (window behind you) kills detection
- **Distance:** Keep hands 1-2 feet from the camera
- **Background:** A plain background helps the AI distinguish hands from clutter

#### 2. Game Controller Not Clicking
- **Window Focus:** Ensure the game window is active (clicked on)
- **Admin Rights:** Some games require the script to be run as Administrator to accept simulated inputs
- **Configuration:** Press `R` to re-map the pedal coordinates if you moved the game window

#### 3. Jittery Cursor / Drawing
- **Stability:** Keep your hand steady
- **Confidence:** Open the code and increase `detection_con` to 0.8
- **Smoothening:** The code implements a moving average to smooth movement; do not move too fast

#### 4. Volume Control Not Working
- Ensure you have installed `pycaw` and `comtypes`
- This feature is strictly Windows only
