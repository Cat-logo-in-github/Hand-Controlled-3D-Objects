import cv2
import time
import numpy as np
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image

class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", num_hands=2):
        """
        Initialize MediaPipe HandLandmarker
        :param model_path: path to hand_landmarker.task
        :param num_hands: maximum number of hands to detect
        """
        options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=hand_landmarker.HandLandmarkerOptions.running_mode.VIDEO,
            num_hands=num_hands
        )
        self.landmarker = hand_landmarker.HandLandmarker.create_from_options(options)
        self.num_hands = num_hands
        self.prev_timestamp = 0

    def detect_hands(self, frame):
        """
        Detect hands in a frame
        :param frame: BGR image from OpenCV
        :return: dict with keys 'hands' and 'handedness'
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        self.prev_timestamp = timestamp_ms

        results = self.landmarker.detect_for_video(mp_img, timestamp_ms)

        hands = []
        handedness_list = []

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Convert to numpy array [21,3] with x,y,z in 0-1 normalized
                lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                hands.append(lm_array)

        if results.handedness:
            for hand_side in results.handedness:
                # Take the top category
                handedness_list.append(hand_side[0].category_name)

        return {"hands": hands, "handedness": handedness_list}

    # --------------------------
    # Utility methods
    # --------------------------
    @staticmethod
    def get_fingertip_positions(hand_landmarks):
        """
        Returns key fingertip positions in normalized coordinates (0-1)
        Order: Thumb_tip, Index_tip, Middle_tip, Ring_tip, Pinky_tip
        """
        tips = [4, 8, 12, 16, 20]
        return hand_landmarks[tips, :]

    @staticmethod
    def get_index_tip(hand_landmarks):
        """Return index fingertip [x, y, z]"""
        return hand_landmarks[8]

    @staticmethod
    def get_thumb_tip(hand_landmarks):
        """Return thumb tip [x, y, z]"""
        return hand_landmarks[4]
