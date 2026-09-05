# Reads in images from a folder 
import cv2
import os 
import numpy as np

class ImageReader:
    def __init__(self, image_path):
        self.image_path = image_path
        if image_path is not None: 
            self.images = self.read_images(image_path)

    def get_path_type(self, image_path):
        if os.path.isdir(self.image_path):
            return "directory"
        elif os.path.isfile(self.image_path):
            _, ext = os.path.splitext(self.image_path)
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                return "image"
        else: 
            raise ValueError("Invalid path: {}".format(self.image_path))
        return self.image_path

    def read_images(self, image_path):
        path_type = self.get_path_type(image_path)
        images = []
        if path_type == "directory":
            for filename in os.listdir(image_path):
                file_path = os.path.join(image_path, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                        image = cv2.imread(file_path)
                        if image is not None:
                            images.append(image)
                            
        elif path_type == "image":
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)

        return np.asarray(images)
        