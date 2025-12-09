import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

class FaceDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)
        
        # Face Mesh for detailed landmarks (468 points)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Face Detection for bounding boxes
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
        self.results = None
        self.detection_results = None
        
    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        self.detection_results = self.face_detection.process(img_rgb)
        
        if draw and self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1
                    )
                )
        return img
    
    def find_face_landmarks(self, img, face_no=0, draw=False):
        lm_list = []
        if self.results.multi_face_landmarks:
            if face_no < len(self.results.multi_face_landmarks):
                face = self.results.multi_face_landmarks[face_no]
                
                for id, lm in enumerate(face.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, x, y])
                    
                    if draw:
                        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        
        return lm_list
    
    def get_face_bbox(self, img):
        """Get face bounding box"""
        bboxes = []
        if self.detection_results.detections:
            for detection in self.detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                bboxes.append([x, y, width, height, detection.score[0]])
        
        return bboxes

class AgeGenderDetector:
    def __init__(self):
        """Initialize age and gender detection models"""
        # Paths to model files (download these from OpenCV's repository)
        self.model_available = False
        
        # Model URLs for download:
        # https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        # https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
        # https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt
        # https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel
        # https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_deploy.prototxt
        
        try:
            # Try to load models (create 'models' folder and download these files)
            self.face_net = cv2.dnn.readNet('models/deploy.prototxt', 
                                           'models/res10_300x300_ssd_iter_140000.caffemodel')
            self.age_net = cv2.dnn.readNet('models/age_deploy.prototxt', 
                                          'models/age_net.caffemodel')
            self.gender_net = cv2.dnn.readNet('models/gender_deploy.prototxt', 
                                             'models/gender_net.caffemodel')
            
            self.model_available = True
            print("✓ Age/Gender models loaded successfully!")
        except:
            print("⚠ Age/Gender models not found. Using estimation mode.")
            print("Download models from OpenCV repository for better accuracy.")
            self.model_available = False
        
        # Model Mean Values
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        # Age and Gender categories
        self.AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                        '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.GENDER_LIST = ['Male', 'Female']
    
    def estimate_age_gender_fallback(self, lm_list):
        """Fallback estimation using facial landmarks (less accurate)"""
        if len(lm_list) < 400:
            return "Unknown", "Unknown", 0, 0
        
        # Simple heuristic-based estimation (not very accurate but works without models)
        # Calculate face proportions
        
        # Distance between eyes
        left_eye = np.array(lm_list[33][1:3])
        right_eye = np.array(lm_list[263][1:3])
        eye_distance = np.linalg.norm(left_eye - right_eye)
        
        # Face height (forehead to chin)
        forehead = np.array(lm_list[10][1:3])
        chin = np.array(lm_list[152][1:3])
        face_height = np.linalg.norm(forehead - chin)
        
        # Face width
        left_face = np.array(lm_list[234][1:3])
        right_face = np.array(lm_list[454][1:3])
        face_width = np.linalg.norm(left_face - right_face)
        
        # Calculate ratios
        face_ratio = face_height / face_width if face_width > 0 else 1.5
        eye_to_face_ratio = eye_distance / face_width if face_width > 0 else 0.3
        
        # Very rough estimation (for demonstration - not medically accurate!)
        # Gender estimation based on face proportions
        if face_ratio > 1.35:  # Typically longer faces
            gender = "Male"
            gender_conf = 0.6
        else:
            gender = "Female"
            gender_conf = 0.6
        
        # Age estimation based on face structure
        if eye_to_face_ratio > 0.32:
            age_range = "(15-25)"
            age_conf = 0.5
        elif eye_to_face_ratio > 0.28:
            age_range = "(25-35)"
            age_conf = 0.5
        else:
            age_range = "(35-50)"
            age_conf = 0.5
        
        return age_range, gender, age_conf, gender_conf
    
    def detect_age_gender(self, img, face_bbox):
        """Detect age and gender using DNN models"""
        if not self.model_available:
            return None, None, 0, 0
        
        try:
            x, y, w, h = face_bbox[:4]
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            # Extract face ROI
            face_roi = img[y:y+h, x:x+w].copy()
            
            if face_roi.size == 0:
                return None, None, 0, 0
            
            # Prepare blob for the models
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                        self.MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender prediction
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender_idx = gender_preds[0].argmax()
            gender = self.GENDER_LIST[gender_idx]
            gender_conf = gender_preds[0][gender_idx]
            
            # Age prediction
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_idx = age_preds[0].argmax()
            age = self.AGE_LIST[age_idx]
            age_conf = age_preds[0][age_idx]
            
            return age, gender, age_conf, gender_conf
            
        except Exception as e:
            print(f"Error in age/gender detection: {e}")
            return None, None, 0, 0

class FacialFeatureAnalyzer:
    def __init__(self):
        # Key landmark indices for facial features
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYEBROW = [336, 296, 334, 293, 300]
        self.RIGHT_EYEBROW = [70, 63, 105, 66, 107]
        self.LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
        self.LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
        self.NOSE_TIP = [1, 2]
        self.NOSE_BRIDGE = [6, 197, 195, 5]
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
    def get_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def get_eye_aspect_ratio(self, eye_landmarks, lm_list):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        if not lm_list:
            return 0
        
        # Vertical distances
        v1 = self.get_distance(lm_list[eye_landmarks[1]][1:], lm_list[eye_landmarks[5]][1:])
        v2 = self.get_distance(lm_list[eye_landmarks[2]][1:], lm_list[eye_landmarks[4]][1:])
        
        # Horizontal distance
        h = self.get_distance(lm_list[eye_landmarks[0]][1:], lm_list[eye_landmarks[3]][1:])
        
        if h == 0:
            return 0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def get_mouth_aspect_ratio(self, lm_list):
        """Calculate Mouth Aspect Ratio (MAR) for smile/yawn detection"""
        if len(lm_list) < 400:
            return 0
        
        # Vertical distance
        v = self.get_distance(lm_list[13][1:], lm_list[14][1:])
        
        # Horizontal distance
        h = self.get_distance(lm_list[61][1:], lm_list[291][1:])
        
        if h == 0:
            return 0
        
        mar = v / h
        return mar
    
    def detect_emotions(self, lm_list):
        """Simple emotion detection based on facial landmarks"""
        if len(lm_list) < 400:
            return "Unknown", (128, 128, 128)
        
        # Calculate ratios
        left_ear = self.get_eye_aspect_ratio(self.LEFT_EYE, lm_list)
        right_ear = self.get_eye_aspect_ratio(self.RIGHT_EYE, lm_list)
        avg_ear = (left_ear + right_ear) / 2
        
        mar = self.get_mouth_aspect_ratio(lm_list)
        
        # Mouth corners for smile detection
        left_mouth = lm_list[61][2]
        right_mouth = lm_list[291][2]
        center_mouth = lm_list[13][2]
        
        smile_indicator = (left_mouth + right_mouth) / 2 - center_mouth
        
        # Emotion detection logic
        if avg_ear < 0.2:
            return "Eyes Closed", (0, 0, 255)
        elif mar > 0.6:
            return "Surprised/Yawning", (0, 255, 255)
        elif smile_indicator < -5:
            return "Happy/Smiling", (0, 255, 0)
        elif avg_ear < 0.25:
            return "Sleepy", (255, 0, 255)
        else:
            return "Neutral", (255, 255, 255)
    
    def calculate_head_pose(self, lm_list, img_shape):
        """Estimate head pose direction"""
        if len(lm_list) < 400:
            return "Unknown", 0, 0
        
        # Key points for head pose
        nose_tip = np.array(lm_list[1][1:3])
        chin = np.array(lm_list[152][1:3])
        left_eye = np.array(lm_list[33][1:3])
        right_eye = np.array(lm_list[263][1:3])
        left_mouth = np.array(lm_list[61][1:3])
        right_mouth = np.array(lm_list[291][1:3])
        
        # Calculate center
        h, w = img_shape[:2]
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        face_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # Horizontal deviation (yaw)
        horizontal_offset = nose_tip[0] - face_center_x
        yaw = horizontal_offset / (w / 2) * 90  # Approximate angle
        
        # Vertical deviation (pitch)
        vertical_offset = nose_tip[1] - face_center_y
        pitch = vertical_offset / (h / 2) * 60
        
        # Determine direction
        if abs(yaw) < 10 and abs(pitch) < 10:
            direction = "Looking Forward"
        elif yaw < -15:
            direction = "Looking Right"
        elif yaw > 15:
            direction = "Looking Left"
        elif pitch < -15:
            direction = "Looking Up"
        elif pitch > 15:
            direction = "Looking Down"
        else:
            direction = "Slight Turn"
        
        return direction, yaw, pitch

class FaceFilterEffects:
    def __init__(self):
        self.filter_type = "none"
        
    def apply_glasses(self, img, lm_list):
        """Draw virtual glasses"""
        if len(lm_list) < 400:
            return img
        
        # Eye positions
        left_eye_center = lm_list[33][1:3]
        right_eye_center = lm_list[263][1:3]
        
        # Glass dimensions
        eye_distance = int(self.get_distance(left_eye_center, right_eye_center))
        glass_width = int(eye_distance * 0.6)
        
        # Draw left glass
        cv2.ellipse(img, left_eye_center, (glass_width//2, glass_width//3), 
                    0, 0, 360, (0, 0, 0), 3)
        
        # Draw right glass
        cv2.ellipse(img, right_eye_center, (glass_width//2, glass_width//3), 
                    0, 0, 360, (0, 0, 0), 3)
        
        # Draw bridge
        cv2.line(img, left_eye_center, right_eye_center, (0, 0, 0), 3)
        
        return img
    
    def apply_mustache(self, img, lm_list):
        """Draw virtual mustache"""
        if len(lm_list) < 400:
            return img
        
        # Nose and mouth positions
        nose_bottom = lm_list[2][1:3]
        left_mouth = lm_list[61][1:3]
        right_mouth = lm_list[291][1:3]
        
        # Mustache dimensions
        mouth_width = int(self.get_distance(left_mouth, right_mouth))
        
        # Draw mustache
        mustache_y = int((nose_bottom[1] + left_mouth[1]) / 2)
        
        # Left side
        cv2.ellipse(img, (left_mouth[0] + mouth_width//4, mustache_y), 
                    (mouth_width//4, mouth_width//8), 0, 0, 180, (0, 0, 0), -1)
        
        # Right side
        cv2.ellipse(img, (right_mouth[0] - mouth_width//4, mustache_y), 
                    (mouth_width//4, mouth_width//8), 0, 0, 180, (0, 0, 0), -1)
        
        return img
    
    def get_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

class AdvancedFacialDetection:
    def __init__(self):
        self.detector = FaceDetector(min_detection_confidence=0.7)
        self.analyzer = FacialFeatureAnalyzer()
        self.filters = FaceFilterEffects()
        self.age_gender_detector = AgeGenderDetector()
        self.mode = "analysis"  # analysis, mesh, features, filters, detection, age_gender
        
        # Tracking variables
        self.blink_counter = 0
        self.blink_threshold = 0.21
        self.smile_detected = False
        
        # Age/Gender cache (to avoid processing every frame)
        self.age_gender_cache = {}
        self.cache_frame_count = 0
        self.cache_update_interval = 30  # Update every 30 frames
        
    def draw_menu(self, img):
        """Draw mode selection menu"""
        menu_height = 80
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], menu_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        modes = [
            ("1-Analysis", (20, 40), (0, 255, 0) if self.mode == "analysis" else (200, 200, 200)),
            ("2-Mesh", (180, 40), (0, 255, 0) if self.mode == "mesh" else (200, 200, 200)),
            ("3-Features", (300, 40), (0, 255, 0) if self.mode == "features" else (200, 200, 200)),
            ("4-Filters", (450, 40), (0, 255, 0) if self.mode == "filters" else (200, 200, 200)),
            ("5-Detection", (590, 40), (0, 255, 0) if self.mode == "detection" else (200, 200, 200)),
            ("6-Age/Gender", (750, 40), (0, 255, 0) if self.mode == "age_gender" else (200, 200, 200)),
        ]
        
        for text, pos, color in modes:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(img, "Q-Quit | G-Glasses | M-Mustache", 
                   (950, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def analysis_mode(self, img, lm_list):
        """Show comprehensive facial analysis"""
        if len(lm_list) < 400:
            cv2.putText(img, "No face detected", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
        
        # Calculate metrics
        left_ear = self.analyzer.get_eye_aspect_ratio(self.analyzer.LEFT_EYE, lm_list)
        right_ear = self.analyzer.get_eye_aspect_ratio(self.analyzer.RIGHT_EYE, lm_list)
        avg_ear = (left_ear + right_ear) / 2
        
        mar = self.analyzer.get_mouth_aspect_ratio(lm_list)
        emotion, emotion_color = self.analyzer.detect_emotions(lm_list)
        direction, yaw, pitch = self.analyzer.calculate_head_pose(lm_list, img.shape)
        
        # Blink detection
        if avg_ear < self.blink_threshold:
            if not hasattr(self, 'eye_closed'):
                self.eye_closed = True
        else:
            if hasattr(self, 'eye_closed') and self.eye_closed:
                self.blink_counter += 1
                self.eye_closed = False
        
        # Info panel
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 100), (500, 450), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Display information
        y_offset = 130
        cv2.putText(img, "FACIAL ANALYSIS", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y_offset += 40
        cv2.putText(img, f"Eye Openness: {avg_ear:.2f}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 35
        cv2.putText(img, f"Mouth Opening: {mar:.2f}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 35
        cv2.putText(img, f"Emotion: {emotion}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
        
        y_offset += 35
        cv2.putText(img, f"Head Pose: {direction}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        y_offset += 35
        cv2.putText(img, f"Yaw: {yaw:.1f}deg  Pitch: {pitch:.1f}deg", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset += 35
        cv2.putText(img, f"Blinks: {self.blink_counter}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 35
        cv2.putText(img, f"Landmarks: {len(lm_list)}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Visual indicators
        # Eye state
        eye_color = (0, 255, 0) if avg_ear > self.blink_threshold else (0, 0, 255)
        cv2.circle(img, (450, 170), 15, eye_color, -1)
        cv2.putText(img, "Eyes", (410, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mouth state
        mouth_color = (0, 255, 255) if mar > 0.5 else (255, 255, 255)
        cv2.circle(img, (450, 240), 15, mouth_color, -1)
        cv2.putText(img, "Mouth", (405, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def mesh_mode(self, img):
        """Show full face mesh"""
        cv2.putText(img, "Face Mesh Mode - 468 Landmarks", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def features_mode(self, img, lm_list):
        """Highlight specific facial features"""
        if len(lm_list) < 400:
            return
        
        # Draw eyes
        for idx in self.analyzer.LEFT_EYE:
            cv2.circle(img, lm_list[idx][1:3], 2, (0, 255, 0), -1)
        for idx in self.analyzer.RIGHT_EYE:
            cv2.circle(img, lm_list[idx][1:3], 2, (0, 255, 0), -1)
        
        # Draw eyebrows
        for idx in self.analyzer.LEFT_EYEBROW:
            cv2.circle(img, lm_list[idx][1:3], 2, (255, 0, 0), -1)
        for idx in self.analyzer.RIGHT_EYEBROW:
            cv2.circle(img, lm_list[idx][1:3], 2, (255, 0, 0), -1)
        
        # Draw lips
        for idx in self.analyzer.LIPS_OUTER:
            cv2.circle(img, lm_list[idx][1:3], 2, (0, 0, 255), -1)
        
        # Draw nose
        for idx in self.analyzer.NOSE_BRIDGE:
            cv2.circle(img, lm_list[idx][1:3], 2, (255, 255, 0), -1)
        
        # Legend
        cv2.putText(img, "Eyes: Green | Eyebrows: Blue", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Lips: Red | Nose: Yellow", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def filters_mode(self, img, lm_list):
        """Apply face filters"""
        if len(lm_list) < 400:
            cv2.putText(img, "No face detected for filters", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
        
        if self.filters.filter_type == "glasses":
            img = self.filters.apply_glasses(img, lm_list)
        elif self.filters.filter_type == "mustache":
            img = self.filters.apply_mustache(img, lm_list)
        elif self.filters.filter_type == "both":
            img = self.filters.apply_glasses(img, lm_list)
            img = self.filters.apply_mustache(img, lm_list)
        
        cv2.putText(img, f"Filter: {self.filters.filter_type.upper()}", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def detection_mode(self, img, bboxes):
        """Show face detection with bounding boxes"""
        for bbox in bboxes:
            x, y, w, h, conf = bbox
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(img, f"{int(conf * 100)}%", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(img, f"Faces Detected: {len(bboxes)}", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def age_gender_mode(self, img, lm_list, bboxes):
        """Age and Gender estimation mode"""
        if len(bboxes) == 0:
            cv2.putText(img, "No face detected", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            return
        
        # Update cache periodically
        self.cache_frame_count += 1
        
        for idx, bbox in enumerate(bboxes):
            x, y, w, h, conf = bbox
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Get or compute age/gender
            cache_key = f"face_{idx}"
            
            if (self.cache_frame_count % self.cache_update_interval == 0 or 
                cache_key not in self.age_gender_cache):
                
                if self.age_gender_detector.model_available:
                    # Use DNN models
                    age, gender, age_conf, gender_conf = self.age_gender_detector.detect_age_gender(
                        img, [x, y, w, h]
                    )
                else:
                    # Use fallback estimation
                    age, gender, age_conf, gender_conf = self.age_gender_detector.estimate_age_gender_fallback(
                        lm_list
                    )
                
                if age and gender:
                    self.age_gender_cache[cache_key] = {
                        'age': age,
                        'gender': gender,
                        'age_conf': age_conf,
                        'gender_conf': gender_conf
                    }
            
            # Display cached or new results
            if cache_key in self.age_gender_cache:
                data = self.age_gender_cache[cache_key]
                age = data['age']
                gender = data['gender']
                age_conf = data['age_conf']
                gender_conf = data['gender_conf']
                
                # Info box background
                info_y = y - 80 if y > 100 else y + h + 10
                overlay = img.copy()
                cv2.rectangle(overlay, (x, info_y), (x + w, info_y + 75), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                
                # Gender
                gender_color = (255, 0, 255) if gender == "Female" else (255, 150, 0)
                cv2.putText(img, f"Gender: {gender}", (x + 5, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, gender_color, 2)
                
                if self.age_gender_detector.model_available:
                    cv2.putText(img, f"Confidence: {gender_conf:.2f}", (x + 5, info_y + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Age
                cv2.putText(img, f"Age: {age}", (x + 5, info_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Model status indicator
                status_text = "DNN Model" if self.age_gender_detector.model_available else "Estimation"
                status_color = (0, 255, 0) if self.age_gender_detector.model_available else (0, 165, 255)
                cv2.putText(img, status_text, (x + 5, info_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, status_color, 1)
        
        # Display mode info
        mode_info = "Using DNN Models" if self.age_gender_detector.model_available else "Using Landmark Estimation"
        info_color = (0, 255, 0) if self.age_gender_detector.model_available else (0, 165, 255)
        
        cv2.putText(img, f"Age/Gender Detection - {mode_info}", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)
        
        if not self.age_gender_detector.model_available:
            cv2.putText(img, "Note: Download DNN models for better accuracy", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        
        prev_time = 0
        
        print("=" * 60)
        print("ADVANCED FACIAL FEATURE DETECTION SYSTEM")
        print("=" * 60)
        print("Controls:")
        print("  1 - Analysis Mode (Emotions, Eye tracking, Head pose)")
        print("  2 - Mesh Mode (Full 468-point face mesh)")
        print("  3 - Features Mode (Eyes, Lips, Nose highlighting)")
        print("  4 - Filters Mode (Virtual glasses, mustache)")
        print("  5 - Detection Mode (Face bounding boxes)")
        print("  6 - Age/Gender Mode (Age and gender estimation)")
        print("  G - Toggle Glasses")
        print("  M - Toggle Mustache")
        print("  Q - Quit")
        print("=" * 60)
        
        if self.age_gender_detector.model_available:
            print("✓ Age/Gender DNN models loaded - High accuracy mode")
        else:
            print("⚠ Using fallback estimation mode")
            print("  Download models for better accuracy:")
            print("  https://github.com/opencv/opencv_3rdparty")
        print("=" * 60)
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            
            # Get face data
            if self.mode == "mesh":
                img = self.detector.find_faces(img, draw=True)
            else:
                img = self.detector.find_faces(img, draw=False)
            
            lm_list = self.detector.find_face_landmarks(img)
            bboxes = self.detector.get_face_bbox(img)
            
            # Draw menu
            self.draw_menu(img)
            
            # Mode-specific processing
            if self.mode == "analysis":
                self.analysis_mode(img, lm_list)
            elif self.mode == "mesh":
                self.mesh_mode(img)
            elif self.mode == "features":
                self.features_mode(img, lm_list)
            elif self.mode == "filters":
                self.filters_mode(img, lm_list)
            elif self.mode == "detection":
                self.detection_mode(img, bboxes)
            elif self.mode == "age_gender":
                self.age_gender_mode(img, lm_list, bboxes)
            
            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(img, f'FPS: {int(fps)}', (img.shape[1] - 150, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow("Advanced Facial Detection", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.mode = "analysis"
            elif key == ord('2'):
                self.mode = "mesh"
            elif key == ord('3'):
                self.mode = "features"
            elif key == ord('4'):
                self.mode = "filters"
            elif key == ord('5'):
                self.mode = "detection"
            elif key == ord('6'):
                self.mode = "age_gender"
                self.cache_frame_count = 0  # Reset cache when entering mode
            elif key == ord('g'):
                if self.filters.filter_type == "glasses":
                    self.filters.filter_type = "none"
                elif self.filters.filter_type == "mustache":
                    self.filters.filter_type = "both"
                else:
                    self.filters.filter_type = "glasses"
            elif key == ord('m'):
                if self.filters.filter_type == "mustache":
                    self.filters.filter_type = "none"
                elif self.filters.filter_type == "glasses":
                    self.filters.filter_type = "both"
                else:
                    self.filters.filter_type = "mustache"
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AdvancedFacialDetection()
    app.run()