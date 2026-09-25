import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import blob_dog
from math import sqrt

class DoGFeatureDetector:
    def __init__(self, min_sigma=1.0, max_sigma=30.0, threshold=0.1):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold

    def convert_to_grayscale(self, image):
        return rgb2gray(image) if image.ndim == 3 else image

    def detect(self, image_gray):
        # Ensure the image is grayscale
        blobs_dog = blob_dog(image_gray, min_sigma=self.min_sigma, max_sigma=self.max_sigma, threshold=self.threshold)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
        return blobs_dog

    def detect_points(self, gray):
        image = self.convert_to_grayscale(gray)
        if image.max() > 1:
            image = image.astype(np.float32) / 255
        blobs = self.detect(image)
        return blobs[:, [1, 0]].astype(np.float32), np.ones(len(blobs), np.float32)

    def visualize(self, image, blobs_dog):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.set_title('Difference of Gaussians (DoG) - Blob Detector')

        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            ax.add_patch(c)

        plt.axis('off')
        plt.show()
