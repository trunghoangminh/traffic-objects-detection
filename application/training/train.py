from base_train import BaseTraining


# This class use for predicting image
class Training(BaseTraining):
    # Constructor
    def __init__(self, image_size, label_size, channel, model_path_name, training_datas, training_labels):
        BaseTraining.__init__(self, image_size, label_size, channel, model_path_name)
        self.training_datas = training_datas
        self.training_labels = training_labels

    def train(self, amount_of_steps):
        print "Starting training ..."
        self.googlenet.train(self.training_datas, self.training_labels, amount_of_steps)
        print "Training done!"

    def save(self):
        print "Starting save model..."
        self.googlenet.save(self.get_model_path_name())
        print "Save model done!"
