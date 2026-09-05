# Calibrate a camera taking in a folder of images
import cv2
import numpy as np
from .ImageReader import ImageReader

class CalibrateCamera:
    def __init__(self, images=None, image_dir=None, chessboard_size=(7, 10), square_size=1.0):
        if images is None:
            images = ImageReader(image_dir, grayscale=True).images
        self.objpoints = []
        self.imgpoints = []
        self.image_size = None
        objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            if self.image_size is None:
                self.image_size = gray.shape[::-1]
            if gray.shape[::-1] != self.image_size:
                raise ValueError("All images must have the same resolution.")
            found, corners = cv2.findChessboardCorners(gray, chessboard_size)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
        print(f"Chessboard found in {len(self.imgpoints)} of {len(images)} images.")

    def calibrate_camera(self, num_images=None, indices=None):
        if indices is None:
            indices = np.arange(len(self.imgpoints))
        if num_images is not None:
            if num_images > len(indices):
                raise ValueError(f"Need {num_images} usable images; only {len(indices)} available.")
            indices = indices[:num_images]
        if len(indices) < 2:
            raise ValueError("Need at least two images with detected corners.")
        return cv2.calibrateCamera(
            [self.objpoints[i] for i in indices],
            [self.imgpoints[i] for i in indices], self.image_size, None, None,
            flags=cv2.CALIB_USE_LU)
