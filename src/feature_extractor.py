# src/feature_extractor.py
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

class MediaPipeHandExtractor:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, image_bgr):
        """
        Input:
            image_bgr: BGR image (numpy array, HxWx3)
        Returns:
            flat vector shape (max_hands * 21 * 3,) with normalized coordinates in [0,1]
            If fewer hands detected, remaining values are zeros.
        """
        if image_bgr is None:
            raise ValueError("image is None")
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        max_hands = 2
        num_landmarks = 21
        kp = np.zeros((max_hands, num_landmarks, 3), dtype=np.float32)

        if results.multi_hand_landmarks:
            # results.multi_hand_landmarks order is arbitrary; keep as-detected order
            for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if h_idx >= max_hands:
                    break
                for l_idx, lm in enumerate(hand_landmarks.landmark):
                    kp[h_idx, l_idx, 0] = lm.x  # normalized x
                    kp[h_idx, l_idx, 1] = lm.y  # normalized y
                    kp[h_idx, l_idx, 2] = lm.z  # roughly depth (negative toward camera)
        # flatten to 1D
        return kp.reshape(-1)

    def close(self):
        self.hands.close()
