#!/usr/bin/env python
from network.googlenet import GoogleNet


# This is base for train and predict extends
class BaseTraining:
    # Constructor
    def __init__(self, image_size, label_size, channel, model_path_name):
        self.image_size = image_size
        self.label_size = label_size
        self.channel = channel
        self.model_path_name = model_path_name
        self.googlenet = GoogleNet(img_size=self.image_size, label_size=self.label_size, channels=self.channel)

    def get_model_path_name(self):
        return self.model_path_name
