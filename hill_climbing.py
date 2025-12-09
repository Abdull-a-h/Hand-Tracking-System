import cv2
import mediapipe as mp
import time
import math
import base64
import pyautogui
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

# Disable PyAutoGUI failsafe for smoother control
pyautogui.FAILSAFE = False

class ControlMapper:
    """Maps game controls from exported configuration"""
    
    def __init__(self):
        self.control_type = 'keyboard'  # 'keyboard' or 'mouse'
        self.controls = {
            'gas': Key.up,
            'brake': Key.down,
            'lean_forward': Key.right,
            'lean_backward': Key.left,
            'special': Key.space
        }
        # Mouse positions for pedals (will be configured)
        self.mouse_positions = {
            'gas': None,  # (x, y) coordinates
            'brake': None
        }
    
    def import_controls(self, control_string):
        """Import controls from GPG-CONTROLS string"""
        try:
            # Decode base64
            decoded = base64.b64decode(control_string.split(':')[1])
            print(f"✓ Control string decoded ({len(decoded)} bytes)")
            
            # For now, let user manually map controls
            print("\n⚠️  Automatic parsing not implemented.")
            print("Please manually configure your controls below.")
            return False
        except Exception as e:
            print(f"✗ Error decoding controls: {e}")
            return False
    
    def configure_manually(self):
        """Manual control configuration"""
        print("\n" + "="*60)
        print("MANUAL CONTROL CONFIGURATION")
        print("="*60)
        print("Press the key you want to use for each action")
        print("(or press ENTER to use default)\n")
        
        actions = [
            ('gas', 'Gas/Accelerate', Key.up),
            ('brake', 'Brake', Key.down),
            ('lean_forward', 'Lean Forward (optional)', Key.right),
            ('lean_backward', 'Lean Backward (optional)', Key.left),
            ('special', 'Special Action (optional)', Key.space)
        ]
        
        for action_key, action_name, default_key in actions:
            print(f"\n{action_name} (default: {default_key}):")
            print("  1. Press W for 'w' key")
            print("  2. Press ↑ for up arrow")
            print("  3. Press ↓ for down arrow")
            print("  4. Press ← for left arrow")
            print("  5. Press → for right arrow")
            print("  6. Press SPACE for space")
            print("  7. Press ENTER to use default")
            
            choice = input("Enter key: ").strip().lower()
            
            if choice == '':
                self.controls[action_key] = default_key
            elif choice == 'w':
                self.controls[action_key] = 'w'
            elif choice == 'a':
                self.controls[action_key] = 'a'
            elif choice == 's':
                self.controls[action_key] = 's'
            elif choice == 'd':
                self.controls[action_key] = 'd'
            elif choice in ['up', '↑']:
                self.controls[action_key] = Key.up
            elif choice in ['down', '↓']:
                self.controls[action_key] = Key.down
            elif choice in ['left', '←']:
                self.controls[action_key] = Key.left
            elif choice in ['right', '→']:
                self.controls[action_key] = Key.right
            elif choice in ['space', ' ']:
                self.controls[action_key] = Key.space
            else:
                # Try to use as character key
                self.controls[action_key] = choice[0] if choice else default_key
        
        print("\n✓ Controls configured!")
        self.print_controls()
        return True
    
    def configure_mouse_controls(self):
        """Configure mouse click positions for gas and brake pedals"""
        print("\n" + "="*60)
        print("MOUSE CONTROL CONFIGURATION")
        print("="*60)
        print("\nYou need to set the positions of your gas and brake pedals.")
        print("\nINSTRUCTIONS:")
        print("1. Open your Hill Climb Racing game")
        print("2. Position it where you'll play")
        print("3. Come back here and follow the prompts")
        print("4. When asked, hover your mouse over the pedal and press ENTER")
        print("\nPress ENTER when your game is ready...")
        input()
        
        # Get gas pedal position
        print("\n🎯 GAS PEDAL:")
        print("   Hover your mouse over the GAS pedal (right side)")
        print("   Press ENTER to capture position...")
        input()
        self.mouse_positions['gas'] = pyautogui.position()
        print(f"   ✓ Gas pedal position: {self.mouse_positions['gas']}")
        
        time.sleep(0.5)
        
        # Get brake pedal position
        print("\n🎯 BRAKE PEDAL:")
        print("   Hover your mouse over the BRAKE pedal (left side)")
        print("   Press ENTER to capture position...")
        input()
        self.mouse_positions['brake'] = pyautogui.position()
        print(f"   ✓ Brake pedal position: {self.mouse_positions['brake']}")
        
        self.control_type = 'mouse'
        
        print("\n✓ Mouse controls configured!")
        print(f"   Gas:   {self.mouse_positions['gas']}")
        print(f"   Brake: {self.mouse_positions['brake']}")
        
        return True
        """Quick configuration with common presets"""
        key_map = {
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'space': Key.space,
            'w': 'w',
            'a': 'a',
            's': 's',
            'd': 'd'
        }
        
        self.controls['gas'] = key_map.get(gas_key.lower(), Key.up)
        self.controls['brake'] = key_map.get(brake_key.lower(), Key.down)
        
        print("✓ Quick controls configured!")
        self.print_controls()
        return True
    
    def print_controls(self):
        """Display current control mapping"""
        print("\nCurrent Control Mapping:")
        print("-" * 40)
        for action, key in self.controls.items():
            print(f"  {action:15} -> {key}")
        print("-" * 40)

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7):
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
    
    def find_position(self, img, hand_no=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
        
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
    
    def get_hand_angle(self, lm_list):
        """Get hand tilt angle for steering"""
        if len(lm_list) < 21:
            return 0
        wrist = lm_list[0][1:3]
        middle_base = lm_list[9][1:3]
        angle = math.degrees(math.atan2(middle_base[1] - wrist[1], middle_base[0] - wrist[0]))
        return angle

class GameController:
    def __init__(self, control_mapper):
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.mapper = control_mapper
        self.keys_pressed = set()
        self.mouse_buttons_pressed = set()
        
    def press_key(self, key):
        """Press and hold a key"""
        if key not in self.keys_pressed:
            self.keyboard.press(key)
            self.keys_pressed.add(key)
    
    def release_key(self, key):
        """Release a key"""
        if key in self.keys_pressed:
            self.keyboard.release(key)
            self.keys_pressed.remove(key)
    
    def click_and_hold(self, position):
        """Click and hold at a specific position"""
        if position not in self.mouse_buttons_pressed:
            pyautogui.mouseDown(position[0], position[1], button='left')
            self.mouse_buttons_pressed.add(position)
    
    def release_click(self, position):
        """Release click at a specific position"""
        if position in self.mouse_buttons_pressed:
            pyautogui.mouseUp(position[0], position[1], button='left')
            self.mouse_buttons_pressed.discard(position)
    
    def release_all_keys(self):
        """Release all currently pressed keys"""
        for key in list(self.keys_pressed):
            self.keyboard.release(key)
        self.keys_pressed.clear()
    
    def release_all_mouse(self):
        """Release all mouse buttons"""
        for position in list(self.mouse_buttons_pressed):
            pyautogui.mouseUp(position[0], position[1], button='left')
        self.mouse_buttons_pressed.clear()
    
    def control_hillclimb(self, gas, brake):
        """Control Hill Climb Racing with custom key or mouse mapping"""
        if self.mapper.control_type == 'mouse':
            # Mouse control mode
            if gas and self.mapper.mouse_positions['gas']:
                self.click_and_hold(self.mapper.mouse_positions['gas'])
            else:
                if self.mapper.mouse_positions['gas']:
                    self.release_click(self.mapper.mouse_positions['gas'])
            
            if brake and self.mapper.mouse_positions['brake']:
                self.click_and_hold(self.mapper.mouse_positions['brake'])
            else:
                if self.mapper.mouse_positions['brake']:
                    self.release_click(self.mapper.mouse_positions['brake'])
        else:
            # Keyboard control mode
            if gas:
                self.press_key(self.mapper.controls['gas'])
            else:
                self.release_key(self.mapper.controls['gas'])
            
            if brake:
                self.press_key(self.mapper.controls['brake'])
            else:
                self.release_key(self.mapper.controls['brake'])

class GestureGameController:
    def __init__(self):
        self.detector = HandDetector()
        self.mapper = ControlMapper()
        self.controller = None
        self.active = False
        self.configured = False
        
    def setup_controls(self):
        """Setup control configuration"""
        print("\n" + "="*60)
        print("CONTROL CONFIGURATION")
        print("="*60)
        print("\nChoose configuration method:")
        print("1. Mouse Control (Click on pedals) - RECOMMENDED for your game")
        print("2. Keyboard - Arrow Keys")
        print("3. Keyboard - WASD Keys")
        print("4. Manual Keyboard Configuration")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            self.configured = self.mapper.configure_mouse_controls()
        elif choice == '2':
            self.mapper.quick_configure('up', 'down')
            self.configured = True
        elif choice == '3':
            self.mapper.quick_configure('w', 's')
            self.configured = True
        elif choice == '4':
            self.configured = self.mapper.configure_manually()
        else:
            print("Invalid choice. Using mouse control (recommended)")
            self.configured = self.mapper.configure_mouse_controls()
        
        if self.configured:
            self.controller = GameController(self.mapper)
    
    def draw_status(self, img, gesture_info):
        # Status panel
        status_color = (0, 255, 0) if self.active else (0, 165, 255)
        status_text = "ACTIVE - CONTROLLING GAME" if self.active else "PAUSED - Press SPACE to start"
        
        cv2.rectangle(img, (10, 10), (img.shape[1] - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (img.shape[1] - 10, 100), status_color, 3)
        
        cv2.putText(img, "Mode: HILL CLIMB RACING", (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, status_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Control mapping display
        if self.mapper.control_type == 'mouse':
            controls_text = f"Mouse Control: Gas {self.mapper.mouse_positions['gas']} | Brake {self.mapper.mouse_positions['brake']}"
        else:
            controls_text = f"Gas: {self.mapper.controls['gas']} | Brake: {self.mapper.controls['brake']}"
        cv2.putText(img, controls_text, (20, img.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Gesture info panel
        if gesture_info:
            y_start = 120
            cv2.rectangle(img, (10, y_start), (450, y_start + len(gesture_info) * 35 + 20), 
                         (0, 0, 0), -1)
            cv2.rectangle(img, (10, y_start), (450, y_start + len(gesture_info) * 35 + 20), 
                         (255, 255, 255), 2)
            
            for i, info in enumerate(gesture_info):
                cv2.putText(img, info, (20, y_start + 30 + i * 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def process_hillclimb(self, lm_list, fingers, angle):
        gas = False
        brake = False
        gesture_info = []
        
        if len(lm_list) > 0:
            # Open palm (4 or 5 fingers up) to accelerate
            if fingers.count(1) >= 4:
                gas = True
                gesture_info.append("⛽ GAS: ON (Open palm detected)")
            else:
                gesture_info.append("⛽ GAS: OFF")
            
            # Fist to brake
            if fingers.count(1) == 0:
                brake = True
                gesture_info.append("🛑 BRAKE: ON (Fist detected)")
            else:
                gesture_info.append("🛑 BRAKE: OFF")
            
            gesture_info.append(f"👆 Fingers up: {fingers.count(1)}/5")
        
        if self.active:
            self.controller.control_hillclimb(gas, brake)
        
        return gesture_info
    
    def run(self):
        if not self.configured:
            self.setup_controls()
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        
        print("\n" + "="*60)
        print("HAND GESTURE GAME CONTROLLER - READY")
        print("="*60)
        print("CONTROLS:")
        print("  SPACE - Start/Stop gesture control")
        print("  R     - Reconfigure controls")
        print("  Q     - Quit")
        print("\nGESTURES:")
        print("  Open PALM (4-5 fingers) -> Gas")
        print("  Make a FIST (0 fingers) -> Brake")
        print("="*60)
        print("\n⚠️  IMPORTANT: Keep your game window visible during play!")
        print("             The script will click on the pedal positions you set.")
        
        prev_time = time.time()
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            
            # Process hand detection
            img = self.detector.find_hands(img)
            lm_list = self.detector.find_position(img)
            
            gesture_info = []
            
            if len(lm_list) > 0:
                fingers = self.detector.fingers_up(lm_list)
                angle = self.detector.get_hand_angle(lm_list)
                gesture_info = self.process_hillclimb(lm_list, fingers, angle)
            else:
                gesture_info = ["❌ NO HAND DETECTED"]
                if self.active:
                    self.controller.release_all_keys()
                    self.controller.release_all_mouse()
            
            self.draw_status(img, gesture_info)
            
            # FPS
            fps = 1 / max(dt, 0.001)
            cv2.putText(img, f'FPS: {int(fps)}', (img.shape[1] - 150, img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow("Hand Gesture Controller", img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to toggle active
                self.active = not self.active
                if not self.active and self.controller:
                    self.controller.release_all_keys()
                    self.controller.release_all_mouse()
                print(f"\n{'🟢 CONTROL ACTIVATED' if self.active else '🔴 CONTROL PAUSED'}")
                if self.active:
                    print("⚠️  Make sure your game window is visible!")
            elif key == ord('r'):  # Reconfigure
                self.active = False
                if self.controller:
                    self.controller.release_all_keys()
                    self.controller.release_all_mouse()
                self.setup_controls()
        
        if self.controller:
            self.controller.release_all_keys()
            self.controller.release_all_mouse()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureGameController()
    app.run()