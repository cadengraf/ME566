import cv2
import numpy as np

class HarrisFeatureDetector:
    def __init__(self, block_size=2, ksize=3, k=0.04):
        self.block_size = block_size
        self.ksize = ksize
        self.k = k

    def detect_corners(self, gray):
        return cv2.cornerHarris(src=np.float32(gray), blockSize=self.block_size,
                                ksize=self.ksize, k=self.k)

    def detect_points(self, gray, threshold=0.01):
        response = self.detect_corners(gray)
        maxima = response == cv2.dilate(response, None)
        ys, xs = np.where(maxima & (response > threshold * response.max()))
        return np.column_stack((xs, ys)).astype(np.float32), response[ys, xs]

    def draw_corners(self, image, dst, threshold=0.01):
        image[dst > threshold * dst.max()] = [0, 0, 255]
        return image
