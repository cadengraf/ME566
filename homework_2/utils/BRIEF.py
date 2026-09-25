"""BRIEF descriptor at supplied keypoints."""
import cv2


class BRIEF:
    norm = cv2.NORM_HAMMING

    def __init__(self, **kwargs):
        if not hasattr(cv2, "xfeatures2d"):
            raise RuntimeError("BRIEF requires opencv-contrib-python")
        self.engine = cv2.xfeatures2d.BriefDescriptorExtractor_create(**kwargs)

    def compute(self, gray, keypoints):
        return self.engine.compute(gray, keypoints)
