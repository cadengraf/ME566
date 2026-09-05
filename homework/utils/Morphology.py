# Compare morphology masks, sizes and operation orders.
from pathlib import Path
import cv2
from .ShowResults import show_results

class Morphology:
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def compare(self):
        for name, (image, simple, adaptive) in self.segmentation.items():
            for method, mask in [("global", simple), ("adaptive", adaptive)]:
                for shape_name, shape in [("rectangle", cv2.MORPH_RECT), ("ellipse", cv2.MORPH_ELLIPSE)]:
                    for size in [3, 5]:
                        kernel = cv2.getStructuringElement(shape, (size, size))
                        eroded = cv2.erode(mask, kernel, iterations=1)
                        dilated = cv2.dilate(mask, kernel, iterations=1)
                        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        open_close = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
                        close_open = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
                        show_results([image, mask, eroded, dilated, opened, closed, open_close, close_open],
                                     ["Original", "Threshold", "Erosion", "Dilation", "Opening", "Closing",
                                      "Open then close", "Close then open"],
                                     f"{Path(name).stem}_{method}_{shape_name}_{size}_morphology")

