from scipy.misc import imread, imresize
import numpy as np


# This class is base image processor
class ImageProcessor:
    def __init__(self, image_size):
        self.image_size = image_size

    # Read image from path
    def read_image(self, full_path):
        return imread(full_path).astype(np.float32)

    # Resize image based on with and heihgt
    def resize_image(self, image_data):
        return imresize(image_data, (self.image_size, self.image_size)).astype(np.float32)
