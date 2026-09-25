# Fast feature detector 
import cv2 

# Want to be able to detect keypoints and specify parameters like threshold
class FAST:
    def __init__(self, threshold, nonmaxSupression):
        self.threshold = threshold
        self.nonmaxSupression = nonmaxSupression
        self.fast = cv2.FastFeatureDetector_create(threshold=self.threshold, nonmaxSupression=self.nonmaxSupression)

    def detect(self, image, mask=None):
        keypoints = self.fast.detect(image, mask)
        return keypoints

    def draw_keypoints(self, image, keypoints):
        output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255,0))
        return output_image

