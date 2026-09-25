import cv2
import numpy as np

class HarrisFeatureDetector:
    def __init__(self, block_size=2, ksize=3, k=0.04):
        self.block_size = block_size
        self.ksize = ksize
        self.k = k

    def detect_corners(self, gray):
        dst = cv2.cornerHarris(src=gray, blockSize=self.block_size, ksize=self.ksize, k=self.k)
        dst = cv2.dilate(dst, None)
        return dst

    def draw_corners(self, image, dst, threshold=0.01):
        image[dst > threshold * dst.max()] = [0, 0, 255]
        return image
    
    