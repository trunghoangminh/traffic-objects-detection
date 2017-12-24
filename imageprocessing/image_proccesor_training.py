from scipy.misc import imread, imresize
import numpy as np
import os

# Handle image input for training
class ImageProcessorTraining:
    def __init__(self, data_set, root_model, image_size, channel, lable_size):
        self.data_set = data_set
        self.root_model = root_model
        self.image_size = image_size
        self.datas = np.empty((0, image_size, image_size, channel))
        self.labels = np.empty((0, lable_size))

    # Produce datas and lales
    def load_data_set(self):
        for key, value in self.data_set.iteritems():
            folder = self.root_model + key;
            image_list = self.get_image_list(folder)
            for image in image_list:
                image_arr = self.read_image(folder + '/' + image)
                image_arr_resize = self.resize_image(image_arr)
                self.append_data(image_arr_resize)
            self.append_label(value)

    # Get images list in sub folder
    def get_image_list(self, folder):
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(folder):
            file_list.extend(filenames)
        return file_list

    # Read image from path
    def read_image(self, path):
        return imread(path).astype(np.float32)

    # Resize image based on with and heihgt
    def resize_image(self, image_data):
        return imresize(image_data, (self.image_size, self.image_size)).astype(np.float32)

    def append_data(self, data):
        self.datas = np.append(self.datas, [data], axis=0)

    def append_label(self, lables):
        self.labels = np.append(self.labels, lables, axis=0)

    # Get maxtrix data training
    def get_datas(self):
        return self.datas

    # Get matrix label training
    def get_labels(self):
        return self.labels
