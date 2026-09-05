# Calibrate a camera taking in a folder of images or a video 

import cv2
import numpy as np
from utils.ImageReader import ImageReader

class CalibrateCamera:
    def __init__(self, images=None, image_dir=None, chessboard_size=None, square_size=None):
        if images is not None: 
            self.images = images
        elif image_dir is not None:
            self.images = ImageReader.read_images(image_dir)

        


    def calibrate_camera(self, folder_images):
        pass


        

    