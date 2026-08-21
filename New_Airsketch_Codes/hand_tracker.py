import cv2
import mediapipe as mp
import math
from collections import deque


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.7
        )

        self.point_history = deque(maxlen=5)

    def smooth_points(self, x, y):
        self.point_history.append((x, y))

        avg_x = int(sum(p[0] for p in self.point_history) / len(self.point_history))
        avg_y = int(sum(p[1] for p in self.point_history) / len(self.point_history))

        return avg_x, avg_y

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]

        self.mp_draw.draw_landmarks(
            frame,
            hand,
            self.mp_hands.HAND_CONNECTIONS
        )

        x1 = int(hand.landmark[8].x * 1280)
        y1 = int(hand.landmark[8].y * 720)

        x2 = int(hand.landmark[12].x * 1280)
        y2 = int(hand.landmark[12].y * 720)

        distance = math.hypot(x2 - x1, y2 - y1)
        pinch = distance < 40

        x1, y1 = self.smooth_points(x1, y1)

        return x1, y1, pinch