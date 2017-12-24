from base_train import BaseTraining


# This class use for predicting image
class Classify(BaseTraining):
    # Constructor
    def __init__(self, image_size, label_size, channel, model_path_name):
        BaseTraining.__init__(self, image_size, label_size, channel, model_path_name)

    def load(self):
        print "Starting load training data ..."
        self.googlenet.load(self.get_model_path_name())
        print "Load done!"

    def predict(self, image_input):
        print "Starting predict..."
        return self.googlenet.predict(image_input)
        print "Predict done!"
