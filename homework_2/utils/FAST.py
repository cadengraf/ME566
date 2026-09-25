"""FAST corner detector."""
import cv2
import numpy as np


class FAST:
    def __init__(self, threshold=20, nonmaxSupression=True):
        self.threshold = threshold
        self.nonmaxSupression = nonmaxSupression
        self.fast = cv2.FastFeatureDetector_create(
            threshold=threshold, nonmaxSuppression=nonmaxSupression)

    def detect(self, image, mask=None):
        return self.fast.detect(image, mask)

    def detect_points(self, image):
        keypoints = self.detect(image)
        points = np.array([point.pt for point in keypoints], np.float32).reshape(-1, 2)
        scores = np.array([point.response for point in keypoints], np.float32)
        return points, scores

    def draw_keypoints(self, image, keypoints):
        output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255,0))
        return output_image

