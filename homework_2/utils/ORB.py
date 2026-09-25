"""ORB descriptor at supplied keypoints."""
import cv2


class ORB:
    norm = cv2.NORM_HAMMING

    def __init__(self, **kwargs):
        self.engine = cv2.ORB_create(**kwargs)

    def compute(self, gray, keypoints):
        return self.engine.compute(gray, keypoints)
