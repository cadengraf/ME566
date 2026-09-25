import cv2 
import numpy as np 

# Shi-Tomasi Corner Detection and address repeatability, speed, number of features to detect. 
class ShiTomasi:
    def __init__(self, max_corners=2000, quality_level=0.01):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = 5
        self.block_size = 3

    def detect_corners(self, gray):
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )
        return (np.empty((0, 2), np.float32) if corners is None
                else corners.reshape(-1, 2))

    def detect_points(self, gray):
        points = self.detect_corners(gray)
        return points, np.ones(len(points), np.float32)

    def draw_corners(self, img, corners):
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (int(x), int(y)), radius=4, color=(0, 255,0), thickness=-1)
        return img

