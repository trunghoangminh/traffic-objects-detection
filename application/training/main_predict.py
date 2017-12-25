from imageprocessing.image_processor_predict import ImageProcessorPredict
from classify import Classify
from utils.app_constants import AppConstants
import math

if __name__ == '__main__':
    image_size = 224
    channel = 3
    label_size = 2

    # Predict
    classify = Classify(image_size, label_size, channel, AppConstants.MODEL_PATH_NAME)
    classify.load()
    image_predict = ImageProcessorPredict(image_size)
    image_arr = image_predict.get_image(AppConstants.ROOT_MODEL + 'car/car1.jpg');

    probs = classify.predict(image_arr)[0]
    cnt = int(sum([math.exp(i + 4) * probs[i] for i in range(len(probs))]))
    probs = [(i, round(100 * p, 1)) for i, p in enumerate(probs)]
    print probs
