import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import time

class GestureSequence:
    """Class to manage gesture sequences"""
    def __init__(self, max_sequence_length=5, timeout=3.0):
        self.max_length = max_sequence_length
        self.timeout = timeout 
        self.sequences = {
            'left': deque(maxlen=max_sequence_length),
            'right': deque(maxlen=max_sequence_length),
            'both': deque(maxlen=max_sequence_length) 
        }
        self.timestamps = {
            'left': deque(maxlen=max_sequence_length),
            'right': deque(maxlen=max_sequence_length),
            'both': deque(maxlen=max_sequence_length)
        }
        
        self.last_match = None
        self.last_match_time = 0
        
    def add_gesture(self, hand_type, gesture):
        """Add a gesture to the sequence"""
        current_time = time.time()

        self._clean_old_gestures(hand_type, current_time)
        if (len(self.sequences[hand_type]) > 0 and 
            self.sequences[hand_type][-1] == gesture):
            return

        self.sequences[hand_type].append(gesture)
        self.timestamps[hand_type].append(current_time)
    
    def add_two_hand_gesture(self, left_gesture, right_gesture):
        """Add a two-hand gesture combination"""
        current_time = time.time()
        combined = f"{left_gesture}+{right_gesture}"

        self._clean_old_gestures('both', current_time)

        if (len(self.sequences['both']) == 0 or 
            self.sequences['both'][-1] != combined):
            self.sequences['both'].append(combined)
            self.timestamps['both'].append(current_time)
    
    def _clean_old_gestures(self, hand_type, current_time):
        """Remove gestures older than timeout"""
        while (len(self.timestamps[hand_type]) > 0 and 
               current_time - self.timestamps[hand_type][0] > self.timeout):
            self.sequences[hand_type].popleft()
            self.timestamps[hand_type].popleft()
    
    def check_patterns(self):
        """Check if current sequences match any defined patterns"""
        matches = []
        
        for hand_type in ['left', 'right']:
            sequence = list(self.sequences[hand_type])
            for pattern_name, pattern in self.patterns.items():
                if self._matches_pattern(sequence, pattern):
                    matches.append((pattern_name, hand_type.upper()))
        
        both_sequence = list(self.sequences['both'])
        for pattern_name, pattern in self.patterns.items():
            if '+' in str(pattern) or pattern_name.endswith('_COMBO') or pattern_name == 'HIGH_FIVE':
                two_hand_pattern = [f"{p}+{p}" for p in pattern] if pattern_name.endswith('_COMBO') else pattern
                if self._matches_pattern(both_sequence, two_hand_pattern):
                    matches.append((pattern_name, 'BOTH'))
        
        if matches:
            return matches[0]  
        return None
    
    def _matches_pattern(self, sequence, pattern):
        """Check if sequence ends with pattern"""
        if len(sequence) < len(pattern):
            return False

        return sequence[-len(pattern):] == pattern
    
    def get_current_sequence(self, hand_type='left'):
        """Get current gesture sequence for display"""
        return list(self.sequences[hand_type])
    
    def clear(self, hand_type=None):
        """Clear sequences"""
        if hand_type:
            self.sequences[hand_type].clear()
            self.timestamps[hand_type].clear()
        else:
            for ht in ['left', 'right', 'both']:
                self.sequences[ht].clear()
                self.timestamps[ht].clear()
class TwoHandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Separate buffers for each hand
        self.left_gesture_buffer = deque(maxlen=5)
        self.right_gesture_buffer = deque(maxlen=5)
        self.left_position_history = deque(maxlen=25)
        self.right_position_history = deque(maxlen=25)
        self.left_dynamic_buffer = deque(maxlen=5)
        self.right_dynamic_buffer = deque(maxlen=5)

        self.sequence_tracker = GestureSequence()

        self.WRIST = 0
        self.THUMB_CMC = 1
        self.THUMB_MCP = 2
        self.THUMB_IP = 3
        self.THUMB_TIP = 4
        self.INDEX_MCP = 5
        self.INDEX_PIP = 6
        self.INDEX_DIP = 7
        self.INDEX_TIP = 8
        self.MIDDLE_MCP = 9
        self.MIDDLE_PIP = 10
        self.MIDDLE_DIP = 11
        self.MIDDLE_TIP = 12
        self.RING_MCP = 13
        self.RING_PIP = 14
        self.RING_DIP = 15
        self.RING_TIP = 16
        self.PINKY_MCP = 17
        self.PINKY_PIP = 18
        self.PINKY_DIP = 19
        self.PINKY_TIP = 20
        
        self.patterns = {
            'UNLOCK': ['PEACE', 'OK'],
            'SELECT': ['PALM', 'FIST'],
            'ZOOM_IN': ['PEACE', 'PALM'],
            'ZOOM_OUT': ['PALM', 'PEACE'],
            'THUMBS_COMBO': ['THUMBS_UP', 'THUMBS_UP'],  
            'PEACE_COMBO': ['PEACE', 'PEACE'], 
            'HIGH_FIVE': ['PALM', 'PALM'],  
        }
    def calculate_distance(self, point1, point2):
        """Calculate 3D Euclidean distance between two points"""
        return math.sqrt(
            (point1[0] - point2[0])**2 + 
            (point1[1] - point2[1])**2 + 
            (point1[2] - point2[2])**2
        )
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points using vectors"""
        v1 = np.array(point1) - np.array(point2)
        v2 = np.array(point3) - np.array(point2)
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def is_finger_extended(self, landmarks, finger_name):
        """Check if a finger is extended using angle-based detection"""
        if finger_name == "thumb":
            angle = self.calculate_angle(
                landmarks[self.THUMB_MCP],
                landmarks[self.THUMB_IP],
                landmarks[self.THUMB_TIP]
            )
            return angle > 160
        
        elif finger_name == "index":
            tip, dip, pip, mcp = self.INDEX_TIP, self.INDEX_DIP, self.INDEX_PIP, self.INDEX_MCP
        elif finger_name == "middle":
            tip, dip, pip, mcp = self.MIDDLE_TIP, self.MIDDLE_DIP, self.MIDDLE_PIP, self.MIDDLE_MCP
        elif finger_name == "ring":
            tip, dip, pip, mcp = self.RING_TIP, self.RING_DIP, self.RING_PIP, self.RING_MCP
        elif finger_name == "pinky":
            tip, dip, pip, mcp = self.PINKY_TIP, self.PINKY_DIP, self.PINKY_PIP, self.PINKY_MCP
        else:
            return False
        
        angle_pip = self.calculate_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        angle_dip = self.calculate_angle(landmarks[pip], landmarks[dip], landmarks[tip])
        
        return angle_pip > 140 and angle_dip > 140
    
    def is_palm_facing_camera(self, landmarks, handedness):
        """Determine if palm is facing camera using cross product and z-depth"""
        wrist = np.array(landmarks[self.WRIST])
        index_mcp = np.array(landmarks[self.INDEX_MCP])
        pinky_mcp = np.array(landmarks[self.PINKY_MCP])
        
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        normal = np.cross(v1, v2)
        palm_facing = normal[2] > 0
        
        if handedness == "Left":
            palm_facing = not palm_facing
        
        return palm_facing
    
    def detect_static_gesture(self, landmarks, palm_facing, handedness):
        """Detect static hand gestures with high accuracy"""
        thumb_extended = self.is_finger_extended(landmarks, "thumb")
        index_extended = self.is_finger_extended(landmarks, "index")
        middle_extended = self.is_finger_extended(landmarks, "middle")
        ring_extended = self.is_finger_extended(landmarks, "ring")
        pinky_extended = self.is_finger_extended(landmarks, "pinky")
        
        fingers_extended = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        num_extended = sum(fingers_extended)
        
        wrist = landmarks[self.WRIST]
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        ring_tip = landmarks[self.RING_TIP]
        pinky_tip = landmarks[self.PINKY_TIP]
        
        if num_extended == 0:
            return "FIST"
        
        if thumb_extended and num_extended==1:
            return "THUMBS_UP"
        
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            finger_separation = self.calculate_distance(index_tip, middle_tip)
            if finger_separation > 0.05 and finger_separation < 0.15:
                return "PEACE"

        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        if thumb_index_dist < 0.04:
            if middle_extended and ring_extended and pinky_extended:
                return "OK"
        if num_extended == 5: 
            return "PALM"

        if index_extended == True:
            direction_vector = np.array(index_tip) - np.array(wrist)
            dx = direction_vector[0]
            dy = direction_vector[1]
            
            angle = math.degrees(math.atan2(-dy, dx))
            if angle < 0:
                angle += 360
            
            if not palm_facing:
                angle = (angle + 180) % 360
            
            sector_size = 45
            half_sector = sector_size / 2
            
            if (angle >= 360 - half_sector) or (angle < half_sector):
                return "RIGHT"
            elif half_sector <= angle < 90 + half_sector:
                return "UP"
            elif 90 + half_sector <= angle < 180 + half_sector:
                return "LEFT"
            elif 180 + half_sector <= angle < 270 + half_sector:
                return "DOWN"
        
        return "UNKNOWN"

    def detect_dynamic_gesture(self, position_history):
        """Detect dynamic gestures with improved accuracy"""
        if len(position_history) < 20:
            return None
        
        positions = list(position_history)
        start_pos = positions[0]
        end_pos = positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        total_distance = math.sqrt(dx**2 + dy**2)
        
        path_length = 0
        for i in range(1, len(positions)):
            path_length += math.sqrt(
                (positions[i][0] - positions[i-1][0])**2 +
                (positions[i][1] - positions[i-1][1])**2
            )
        
        straightness = total_distance / (path_length + 1e-6)

        if total_distance > 0.20 and straightness > 0.7:
            angle = math.degrees(math.atan2(dy, dx))
            
            if -45 <= angle < 45:
                return "SWIPE_RIGHT"
            elif 45 <= angle < 135:
                return "SWIPE_DOWN"
            elif abs(angle) >= 135:
                return "SWIPE_LEFT"
            elif -135 <= angle < -45:
                return "SWIPE_UP"

        if len(positions) >= 18:
            x_positions = [pos[0] for pos in positions]
            direction_changes = 0
            for i in range(1, len(x_positions) - 1):
                if (x_positions[i] > x_positions[i-1] and x_positions[i] > x_positions[i+1]):
                    direction_changes += 1
                elif (x_positions[i] < x_positions[i-1] and x_positions[i] < x_positions[i+1]):
                    direction_changes += 1
            
            x_variance = np.var(x_positions)
            if direction_changes >= 4 and x_variance > 0.004:
                return "WAVE"
        
        return None
    
    def smooth_gesture(self, gesture, hand_buffer):
        """Apply smoothing to reduce jitter"""
        if gesture is None or gesture == "UNKNOWN":
            return None
        
        hand_buffer.append(gesture)
        
        if len(hand_buffer) >= 3:
            from collections import Counter
            most_common = Counter(hand_buffer).most_common(1)[0][0]
            return most_common
        
        return gesture
    
    def detect_two_hand_combo(self, hand_data):
        left_gesture = hand_data['Left']['gesture']
        right_gesture = hand_data['Right']['gesture']
        
        # Both hands must have valid gestures
        if not left_gesture or not right_gesture:
            return None
        
        if left_gesture in ['UNKNOWN', 'NO_HAND'] or right_gesture in ['UNKNOWN', 'NO_HAND']:
            return None
        
        # Check against patterns (assuming patterns are tuples now)
        for pattern_name, (left_pattern, right_pattern) in self.patterns.items():
            if left_gesture == left_pattern and right_gesture == right_pattern:
                return pattern_name
    
        return None

    def process_frame(self, frame):
        """Process frame and detect gestures for both hands"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        # Initialize hand data
        hand_data = {
            'Left': {'gesture': None, 'dynamic': None, 'palm_facing': True, 'confidence': 0.0},
            'Right': {'gesture': None, 'dynamic': None, 'palm_facing': True, 'confidence': 0.0}
        }
        dynamic_gesture = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score

                if hand_label == "Left":
                    landmark_color = (0, 255, 0)  
                    connection_color = (0, 200, 0)
                else:
                    landmark_color = (255, 0, 0) 
                    connection_color = (200, 0, 0)
                
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=connection_color, thickness=2)
                )
                
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                palm_facing = self.is_palm_facing_camera(landmarks, hand_label)
                raw_gesture = self.detect_static_gesture(landmarks, palm_facing, hand_label)
                
                # Get appropriate buffers
                if hand_label == "Left":
                    gesture_buffer = self.left_gesture_buffer
                    position_history = self.left_position_history
                    dynamic_buffer = self.left_dynamic_buffer
                else:
                    gesture_buffer = self.right_gesture_buffer
                    position_history = self.right_position_history
                    dynamic_buffer = self.right_dynamic_buffer

                gesture = self.smooth_gesture(raw_gesture, gesture_buffer)
                if gesture is None:
                    gesture = raw_gesture

                if gesture in ["UNKNOWN", "PALM", None]:
                    middle_tip_pos = (landmarks[self.MIDDLE_TIP][0], landmarks[self.MIDDLE_TIP][1])
                    position_history.append(middle_tip_pos)
    
                    raw_dynamic = self.detect_dynamic_gesture(position_history)
                else:
                    # Clear dynamic buffer when static gesture is active
                    dynamic_buffer.clear()
                    raw_dynamic = None
                if raw_dynamic:
                    dynamic_buffer.append(raw_dynamic)
                    if len(dynamic_buffer) >= 4:
                        from collections import Counter
                        dynamic_gesture = Counter(dynamic_buffer).most_common(1)[0][0]
                else:
                    dynamic_buffer.clear()
                
                # Store hand data
                hand_data[hand_label] = {
                    'gesture': gesture if gesture != "UNKNOWN" else None,
                    'dynamic': dynamic_gesture,
                    'palm_facing': palm_facing,
                    'confidence': confidence
                }

                if gesture and gesture not in ["UNKNOWN", "NO_HAND"]:
                    self.sequence_tracker.add_gesture(hand_label.lower(), gesture)
                if dynamic_gesture:
                    self.sequence_tracker.add_gesture(hand_label.lower(), dynamic_gesture)
        
        else:
            # Clear buffers if no hands detected
            self.left_position_history.clear()
            self.right_position_history.clear()
            self.left_gesture_buffer.clear()
            self.right_gesture_buffer.clear()
            self.left_dynamic_buffer.clear()
            self.right_dynamic_buffer.clear()

        sequence_match = self.detect_two_hand_combo(hand_data)
        self._draw_ui(frame, hand_data, sequence_match)
        
        return frame, hand_data, sequence_match
    
    def _draw_ui(self, frame, hand_data, sequence_match):
        """Draw UI overlay with hand information and sequences"""
        h, w, _ = frame.shape
        
        cv2.rectangle(frame, (5, 5), (350, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (350, 150), (0, 255, 0), 2)
        cv2.putText(frame, "LEFT HAND", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if hand_data['Left']['gesture']:
            cv2.putText(frame, f"Static: {hand_data['Left']['gesture']}", (15, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            orientation = "PALM" if hand_data['Left']['palm_facing'] else "BACK"
            cv2.putText(frame, f"Orient: {orientation}", (15, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "No detection", (15, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        if hand_data['Left']['dynamic']:
            cv2.putText(frame, f"Dynamic: {hand_data['Left']['dynamic']}", (15, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        left_seq = self.sequence_tracker.get_current_sequence('left')
        if left_seq:
            seq_text = " > ".join(left_seq[-3:])  
            cv2.putText(frame, f"Seq: {seq_text}", (15, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)

        cv2.rectangle(frame, (w-355, 5), (w-5, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (w-355, 5), (w-5, 150), (255, 0, 0), 2)
        cv2.putText(frame, "RIGHT HAND", (w-345, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if hand_data['Right']['gesture']:
            cv2.putText(frame, f"Static: {hand_data['Right']['gesture']}", (w-345, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            orientation = "PALM" if hand_data['Right']['palm_facing'] else "BACK"
            cv2.putText(frame, f"Orient: {orientation}", (w-345, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "No detection", (w-345, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        if hand_data['Right']['dynamic']:
            cv2.putText(frame, f"Dynamic: {hand_data['Right']['dynamic']}", (w-345, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        right_seq = self.sequence_tracker.get_current_sequence('right')
        if right_seq:
            seq_text = " > ".join(right_seq[-3:])
            cv2.putText(frame, f"Seq: {seq_text}", (w-345, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
            
        cv2.rectangle(frame, (w//2 - 275, 5), (w//2 + 275, 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (w//2 - 275, 5), (w//2 + 275, 50), (255, 255, 255), 2)

        text = "Press 'q' to exit, 'r' to refresh and 'c' to clear sequence."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  
        
        cv2.rectangle(frame, (w//2 - 275, 55), (w//2 + 275, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (w//2 - 275, 55), (w//2 + 275, 100), (255, 255, 255), 2)
        if sequence_match:
            text = f"COMBO: {sequence_match}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = TwoHandGestureDetector()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame, hand_data, sequence_match = detector.process_frame(frame)
    
    frame_count += 1
    cv2.imshow('Two-Hand Gesture Detection with Sequencing', processed_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        detector.sequence_tracker.clear()
    elif key == ord('c'):
        detector.left_gesture_buffer.clear()
        detector.right_gesture_buffer.clear()
        detector.left_position_history.clear()
        detector.right_position_history.clear()
        detector.sequence_tracker.clear()

cap.release()
cv2.destroyAllWindows()
detector.hands.close()

