"""SIFT descriptor at supplied keypoints."""
import cv2


class SIFT:
    norm = cv2.NORM_L2

    def __init__(self, **kwargs):
        self.engine = cv2.SIFT_create(**kwargs)

    def compute(self, gray, keypoints):
        return self.engine.compute(gray, keypoints)

    def detect_and_compute(self, gray):
        return self.engine.detectAndCompute(gray, None)

