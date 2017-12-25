from image_processor import ImageProcessor


# This class for get image for predict
class ImageProcessorPredict(ImageProcessor):
    def __init__(self, image_size):
        ImageProcessor.__init__(self, image_size)

    # Get image for predict
    def get_image(self, full_path):
        return self.resize_image(self.read_image(full_path))
