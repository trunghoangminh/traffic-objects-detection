import numpy as np

from imageprocessing.image_proccesor_training import ImageProcessorTraining
from train import Training
from utils.app_constants import AppConstants

if __name__ == '__main__':
    image_size = 224
    channel = 3
    label_size = 2
    data_set = {'car': np.array([[1, 0]] * 10), 'motobike': np.array([[0, 1]] * 10)}

    # Prepare data
    image = ImageProcessorTraining(data_set, AppConstants.ROOT_MODEL, image_size, channel, label_size)
    image.load_data_set()

    print image.get_datas()
    # Train
    train = Training(image_size, label_size, channel, AppConstants.MODEL_PATH_NAME, image.get_datas(),
                     image.get_labels())
    train.train(10)
    train.save()

    # classify = Classify(image_size, label_size, channel, AppConstants.MODEL_PATH_NAME)
    # classify.load()
    # print classify.predict(image_result)
    # # cnt = int(sum([math.exp(i + 4) * probs[i] for i in range(len(probs))]))
    # # probs = [(i, round(100 * p, 1)) for i, p in enumerate(probs)]
    # # print probs
