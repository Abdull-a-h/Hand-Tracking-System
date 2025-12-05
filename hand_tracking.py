import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False
    print("Note: Install pycaw for volume control: pip install pycaw")

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )
        return img
    
    def find_position(self, img, hand_no=0, draw=False):
        lm_list = []
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return lm_list
    
    def fingers_up(self, lm_list):
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]
        
        if len(lm_list) != 0:
            # Thumb
            if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Other fingers
            for id in range(1, 5):
                if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return fingers
    
    def get_hand_type(self, hand_no=0):
        if self.results.multi_handedness:
            if hand_no < len(self.results.multi_handedness):
                return self.results.multi_handedness[hand_no].classification[0].label
        return "Unknown"

class GestureRecognizer:
    def __init__(self):
        self.gesture_name = "None"
        self.gesture_color = (255, 255, 255)
        self.prev_gesture = None
        self.gesture_hold_frames = 0
        
    def get_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def recognize_gesture(self, lm_list, fingers):
        fingers_up = fingers.count(1)
        
        thumb_tip = lm_list[4][1:3]
        index_tip = lm_list[8][1:3]
        middle_tip = lm_list[12][1:3]
        ring_tip = lm_list[16][1:3]
        pinky_tip = lm_list[20][1:3]
        wrist = lm_list[0][1:3]
        
        # Gesture recognition
        if fingers_up == 0:
            self.gesture_name = "FIST"
            self.gesture_color = (0, 0, 255)
        elif fingers == [1, 0, 0, 0, 0]:
            if thumb_tip[1] < wrist[1]:
                self.gesture_name = "THUMBS_UP"
                self.gesture_color = (0, 255, 0)
            else:
                self.gesture_name = "THUMBS_DOWN"
                self.gesture_color = (0, 165, 255)
        elif fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
            self.gesture_name = "POINTING"
            self.gesture_color = (255, 255, 0)
        elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
            distance = self.get_distance(index_tip, middle_tip)
            if distance > 40:
                self.gesture_name = "PEACE"
                self.gesture_color = (255, 0, 255)
            else:
                self.gesture_name = "TWO"
                self.gesture_color = (180, 180, 0)
        elif fingers == [0, 1, 1, 1, 0] or fingers == [1, 1, 1, 1, 0]:
            self.gesture_name = "THREE"
            self.gesture_color = (100, 200, 100)
        elif fingers == [0, 1, 1, 1, 1]:
            self.gesture_name = "FOUR"
            self.gesture_color = (200, 100, 200)
        elif fingers_up == 5:
            self.gesture_name = "FIVE"
            self.gesture_color = (0, 255, 255)
        elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:
            self.gesture_name = "ROCK"
            self.gesture_color = (147, 20, 255)
        elif fingers == [1, 0, 0, 0, 1]:
            self.gesture_name = "CALL_ME"
            self.gesture_color = (200, 200, 0)
        elif fingers == [1, 1, 0, 0, 1]:
            self.gesture_name = "SPIDERMAN"
            self.gesture_color = (0, 50, 255)
        else:
            self.gesture_name = f"CUSTOM"
            self.gesture_color = (128, 128, 128)
        
        # Track gesture stability
        if self.gesture_name == self.prev_gesture:
            self.gesture_hold_frames += 1
        else:
            self.gesture_hold_frames = 0
        self.prev_gesture = self.gesture_name
            
        return self.gesture_name, self.gesture_color

class VirtualPainter:
    def __init__(self, canvas_size):
        self.canvas = np.zeros(canvas_size, dtype=np.uint8)
        self.draw_color = (255, 0, 0)
        self.brush_size = 5
        self.prev_pos = None
        self.drawing_enabled = True
        
    def draw(self, pos):
        if self.prev_pos and self.drawing_enabled:
            cv2.line(self.canvas, self.prev_pos, pos, self.draw_color, self.brush_size)
        self.prev_pos = pos
    
    def clear_canvas(self):
        self.canvas = np.zeros(self.canvas.shape, dtype=np.uint8)
        self.prev_pos = None
    
    def set_color(self, color):
        self.draw_color = color
    
    def get_canvas(self):
        return self.canvas

class VolumeController:
    def __init__(self):
        self.volume_available = AUDIO_AVAILABLE
        if self.volume_available:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.vol_range = self.volume.GetVolumeRange()
                self.min_vol = self.vol_range[0]
                self.max_vol = self.vol_range[1]
            except:
                self.volume_available = False
    
    def set_volume_by_distance(self, distance, max_distance=300):
        if not self.volume_available:
            return None
        
        # Map distance to volume
        vol_percentage = np.interp(distance, [30, max_distance], [0, 100])
        vol = np.interp(distance, [30, max_distance], [self.min_vol, self.max_vol])
        
        try:
            self.volume.SetMasterVolumeLevel(vol, None)
            return int(vol_percentage)
        except:
            return None

class AdvancedHandTracking:
    def __init__(self):
        self.detector = HandDetector(detection_con=0.7, track_con=0.7, max_hands=2)
        self.gesture_recognizer = GestureRecognizer()
        self.mode = "tracking"  # tracking, drawing, volume, game
        self.painter = None
        self.volume_controller = VolumeController()
        self.game_score = 0
        self.targets = []
        
    def draw_menu(self, img):
        menu_height = 80
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], menu_height), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        modes = [
            ("1-Tracking", (50, 40), (0, 255, 0) if self.mode == "tracking" else (200, 200, 200)),
            ("2-Drawing", (250, 40), (0, 255, 0) if self.mode == "drawing" else (200, 200, 200)),
            ("3-Volume", (450, 40), (0, 255, 0) if self.mode == "volume" else (200, 200, 200)),
            ("4-Game", (650, 40), (0, 255, 0) if self.mode == "game" else (200, 200, 200)),
        ]
        
        for text, pos, color in modes:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(img, "Press Q to Quit | C to Clear", (900, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def tracking_mode(self, img, lm_list, fingers, gesture_name, gesture_color, hand_type):
        # Info panel
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 100), (450, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        cv2.putText(img, f'Hand: {hand_type}', (30, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'Fingers: {fingers.count(1)}', (30, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'Gesture: {gesture_name}', (30, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
        cv2.putText(img, f'Pattern: {fingers}', (30, 235),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(img, f'Landmarks: {len(lm_list)}', (30, 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Large gesture display
        cv2.putText(img, gesture_name, (img.shape[1]//2 - 200, 150),
                   cv2.FONT_HERSHEY_DUPLEX, 2, gesture_color, 3)
        
        # Highlight fingertips
        tip_ids = [4, 8, 12, 16, 20]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, tip_id in enumerate(tip_ids):
            if fingers[i] == 1:
                cv2.circle(img, (lm_list[tip_id][1], lm_list[tip_id][2]), 10, colors[i], cv2.FILLED)
    
    def drawing_mode(self, img, lm_list, fingers):
        if self.painter is None:
            self.painter = VirtualPainter(img.shape)
        
        # Color palette
        colors = [
            ("Red", (0, 0, 255), 50),
            ("Green", (0, 255, 0), 150),
            ("Blue", (255, 0, 0), 250),
            ("Yellow", (0, 255, 255), 350),
            ("White", (255, 255, 255), 450),
        ]
        
        for name, color, x in colors:
            cv2.rectangle(img, (x, 100), (x+80, 150), color, -1)
            cv2.putText(img, name, (x+5, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Use index finger to draw
        if fingers[1] == 1 and fingers[2] == 0:  # Only index up
            index_pos = (lm_list[8][1], lm_list[8][2])
            
            # Check if selecting color
            if 100 < index_pos[1] < 150:
                for name, color, x in colors:
                    if x < index_pos[0] < x + 80:
                        self.painter.set_color(color)
            else:
                self.painter.draw(index_pos)
            
            cv2.circle(img, index_pos, 10, (0, 255, 0), cv2.FILLED)
        else:
            self.painter.prev_pos = None
        
        # Overlay canvas
        canvas = self.painter.get_canvas()
        img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        
        cv2.putText(img, "Index finger to draw | Peace to pause", 
                   (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img
    
    def volume_mode(self, img, lm_list, fingers):
        # Use thumb and index finger distance for volume
        if fingers[0] == 1 and fingers[1] == 1:
            thumb_pos = lm_list[4][1:3]
            index_pos = lm_list[8][1:3]
            
            cx, cy = (thumb_pos[0] + index_pos[0]) // 2, (thumb_pos[1] + index_pos[1]) // 2
            
            cv2.circle(img, thumb_pos, 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, index_pos, 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, thumb_pos, index_pos, (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            
            distance = self.gesture_recognizer.get_distance(thumb_pos, index_pos)
            vol_percent = self.volume_controller.set_volume_by_distance(distance)
            
            # Volume bar
            vol_bar_height = int(np.interp(distance, [30, 300], [400, 50]))
            cv2.rectangle(img, (50, 50), (100, 450), (0, 255, 0), 3)
            cv2.rectangle(img, (50, vol_bar_height), (100, 450), (0, 255, 0), cv2.FILLED)
            
            if vol_percent is not None:
                cv2.putText(img, f'{vol_percent}%', (120, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(img, 'Volume control unavailable', (120, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(img, f'Distance: {int(distance)}', (120, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def game_mode(self, img, lm_list, fingers):
        # Simple target shooting game
        if len(self.targets) < 5 and np.random.random() > 0.98:
            x = np.random.randint(100, img.shape[1]-100)
            y = np.random.randint(200, img.shape[0]-100)
            self.targets.append([x, y, 30])
        
        # Draw targets
        for target in self.targets:
            cv2.circle(img, (target[0], target[1]), target[2], (0, 0, 255), 3)
            cv2.circle(img, (target[0], target[1]), 5, (0, 0, 255), cv2.FILLED)
        
        # Use index finger to shoot
        if fingers[1] == 1:
            index_pos = (lm_list[8][1], lm_list[8][2])
            cv2.circle(img, index_pos, 20, (0, 255, 0), 3)
            
            # Check collision
            for target in self.targets[:]:
                dist = math.sqrt((target[0]-index_pos[0])**2 + (target[1]-index_pos[1])**2)
                if dist < target[2]:
                    self.targets.remove(target)
                    self.game_score += 10
        
        # Score display
        cv2.putText(img, f'Score: {self.game_score}', (img.shape[1]-200, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, 'Point at targets!', (img.shape[1]-250, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        
        prev_time = 0
        
        print("=" * 60)
        print("ADVANCED HAND TRACKING SYSTEM")
        print("=" * 60)
        print("Controls:")
        print("  1 - Tracking Mode (Gesture Recognition)")
        print("  2 - Drawing Mode (Virtual Painter)")
        print("  3 - Volume Control Mode")
        print("  4 - Game Mode (Target Shooting)")
        print("  C - Clear Canvas/Reset")
        print("  Q - Quit")
        print("=" * 60)
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            img = self.detector.find_hands(img)
            
            # Menu
            self.draw_menu(img)
            
            lm_list = self.detector.find_position(img, hand_no=0)
            
            if len(lm_list) != 0:
                fingers = self.detector.fingers_up(lm_list)
                hand_type = self.detector.get_hand_type(0)
                gesture_name, gesture_color = self.gesture_recognizer.recognize_gesture(lm_list, fingers)
                
                if self.mode == "tracking":
                    self.tracking_mode(img, lm_list, fingers, gesture_name, gesture_color, hand_type)
                elif self.mode == "drawing":
                    img = self.drawing_mode(img, lm_list, fingers)
                elif self.mode == "volume":
                    self.volume_mode(img, lm_list, fingers)
                elif self.mode == "game":
                    self.game_mode(img, lm_list, fingers)
            
            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(img, f'FPS: {int(fps)}', (img.shape[1] - 150, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow("Advanced Hand Tracking", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.mode = "tracking"
            elif key == ord('2'):
                self.mode = "drawing"
            elif key == ord('3'):
                self.mode = "volume"
            elif key == ord('4'):
                self.mode = "game"
            elif key == ord('c'):
                if self.painter:
                    self.painter.clear_canvas()
                self.game_score = 0
                self.targets = []
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AdvancedHandTracking()
    app.run()