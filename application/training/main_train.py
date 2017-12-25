from imageprocessing.image_processor_training import ImageProcessorTraining
from train import Training
import numpy as np
from utils.app_constants import AppConstants

if __name__ == '__main__':
    image_size = 224
    channel = 3
    label_size = 2
    data_set = {'car': np.array([[1, 0]] * 10), 'motobike': np.array([[0, 1]] * 10)}

    # Prepare data
    image = ImageProcessorTraining(data_set, AppConstants.ROOT_MODEL, image_size, channel, label_size)
    image.load_data_set()

    # Train
    train = Training(image_size, label_size, channel, AppConstants.MODEL_PATH_NAME, image.get_datas(),
                     image.get_labels())
    train.train(1000)
    train.save()

