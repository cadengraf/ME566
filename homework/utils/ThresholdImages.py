# Compare global and adaptive thresholding.
from pathlib import Path
import cv2
from .ImageReader import ImageReader
from .ShowResults import show_results

class ThresholdImages:
    def __init__(self, image_dir=None):
        self.image_dir = Path(image_dir) if image_dir is not None else Path(__file__).resolve().parents[2] / "images"

    def compare(self):
        segmentation = {}
        for name in ["coins1.jpg", "coins2.jpg", "screws.jpeg"]:
            image = ImageReader(self.image_dir / name, grayscale=True).images[0]
            smooth = cv2.GaussianBlur(image, (3, 3), 0)
            _, simple = cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY_INV)
            adaptive = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 31, 5)
            segmentation[name] = (image, simple, adaptive)
            show_results([image, smooth, simple, adaptive],
                         ["Original", "Gaussian 3 x 3", "Global: 127", "Adaptive: 31, C=5"],
                         Path(name).stem + "_threshold")
        return segmentation

