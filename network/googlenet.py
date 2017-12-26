"""
Using tflearn and apply GoogleNet model
"""
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn


# This class contains algorithm about GoogleNet and some info for application
class GoogleNet:
    def __init__(self, img_size, label_size, channels):
        # Image input for training
        self.img_size = img_size

        # Label input for training
        self.label_size = label_size

        # Channels input for training
        self.channels = channels

        # GoogleNet algorithm
        self.network = input_data(shape=[None, img_size, img_size, channels], name='data')

        self.conv1_7x7_s2 = conv_2d(self.network, 64, 7, strides=2, activation='relu', name='conv1_7x7_s2')

        self.pool1_3x3_s2 = max_pool_2d(self.conv1_7x7_s2, 3, strides=2, name='pool1_3x3_s2')

        # Special case
        self.pool1_3x3_s2 = local_response_normalization(self.pool1_3x3_s2, name='pool1_norm1')

        self.conv2_3x3_reduce = conv_2d(self.pool1_3x3_s2, 64, 1, activation='relu', name='conv2_3x3_reduce')

        self.conv2_3x3 = conv_2d(self.conv2_3x3_reduce, 192, 3, activation='relu', name='conv2_3x3')

        # Special case
        self.conv2_3x3 = local_response_normalization(self.conv2_3x3, name='conv2_norm2')

        self.pool2_3x3_s2 = max_pool_2d(self.conv2_3x3, kernel_size=3, strides=2, name='pool2_3x3_s2')

        self.inception_3a_1x1 = conv_2d(self.pool2_3x3_s2, 64, 1, activation='relu', name='inception_3a_1x1')

        self.inception_3a_3x3_reduce = conv_2d(self.pool2_3x3_s2, 96, 1, activation='relu',
                                               name='inception_3a_3x3_reduce')

        self.inception_3a_3x3 = conv_2d(self.inception_3a_3x3_reduce, 128, filter_size=3, activation='relu',
                                        name='inception_3a_3x3')

        self.inception_3a_5x5_reduce = conv_2d(self.pool2_3x3_s2, 16, filter_size=1, activation='relu',
                                               name='inception_3a_5x5_reduce')

        self.inception_3a_5x5 = conv_2d(self.inception_3a_5x5_reduce, 32, filter_size=5, activation='relu',
                                        name='inception_3a_5x5')

        self.inception_3a_pool = max_pool_2d(self.pool2_3x3_s2, kernel_size=3, strides=1, name='inception_3a_pool')

        self.inception_3a_pool_proj = conv_2d(self.inception_3a_pool, 32, filter_size=1, activation='relu',
                                              name='inception_3a_pool_proj')
        # Merge the inception 3a
        self.inception_3a_output = merge(
            [self.inception_3a_1x1, self.inception_3a_3x3, self.inception_3a_5x5, self.inception_3a_pool_proj],
            mode='concat', axis=3, name='inception_3a_output')

        self.inception_3b_1x1 = conv_2d(self.inception_3a_output, 128, filter_size=1, activation='relu',
                                        name='inception_3b_1x1')

        self.inception_3b_3x3_reduce = conv_2d(self.inception_3a_output, 128, filter_size=1, activation='relu',
                                               name='inception_3b_3x3_reduce')

        self.inception_3b_3x3 = conv_2d(self.inception_3b_3x3_reduce, 192, filter_size=3, activation='relu',
                                        name='inception_3b_3x3')

        self.inception_3b_5x5_reduce = conv_2d(self.inception_3a_output, 32, filter_size=1, activation='relu',
                                               name='inception_3b_5x5_reduce')

        self.inception_3b_5x5 = conv_2d(self.inception_3b_5x5_reduce, 96, filter_size=5, name='inception_3b_5x5')

        self.inception_3b_pool = max_pool_2d(self.inception_3a_output, kernel_size=3, strides=1,
                                             name='inception_3b_pool')

        self.inception_3b_pool_proj = conv_2d(self.inception_3b_pool, 64, filter_size=1, activation='relu',
                                              name='inception_3b_pool_proj')
        # Merge the inception 3b
        self.inception_3b_output = merge(
            [self.inception_3b_1x1, self.inception_3b_3x3, self.inception_3b_5x5, self.inception_3b_pool_proj],
            mode='concat', axis=3, name='inception_3b_output')

        self.pool3_3x3_s2 = max_pool_2d(self.inception_3b_output, kernel_size=3, strides=2, name='pool3_3x3_s2')

        self.inception_4a_1x1 = conv_2d(self.pool3_3x3_s2, 192, filter_size=1, activation='relu',
                                        name='inception_4a_1x1')

        self.inception_4a_3x3_reduce = conv_2d(self.pool3_3x3_s2, 96, filter_size=1, activation='relu',
                                               name='inception_4a_3x3_reduce')

        self.inception_4a_3x3 = conv_2d(self.inception_4a_3x3_reduce, 208, filter_size=3, activation='relu',
                                        name='inception_4a_3x3')
        self.inception_4a_5x5_reduce = conv_2d(self.pool3_3x3_s2, 16, filter_size=1, activation='relu',
                                               name='inception_4a_5x5_reduce')
        self.inception_4a_5x5 = conv_2d(self.inception_4a_5x5_reduce, 48, filter_size=5, activation='relu',
                                        name='inception_4a_5x5')

        self.inception_4a_pool = max_pool_2d(self.pool3_3x3_s2, kernel_size=3, strides=1, name='inception_4a_pool')

        self.inception_4a_pool_proj = conv_2d(self.inception_4a_pool, 64, filter_size=1, activation='relu',
                                              name='inception_4a_pool_proj')

        self.inception_4a_output = merge(
            [self.inception_4a_1x1, self.inception_4a_3x3, self.inception_4a_5x5, self.inception_4a_pool_proj],
            mode='concat', axis=3, name='inception_4a_output')

        self.inception_4b_1x1 = conv_2d(self.inception_4a_output, 160, filter_size=1, activation='relu',
                                        name='inception_4b_1x1')

        self.inception_4b_3x3_reduce = conv_2d(self.inception_4a_output, 112, filter_size=1, activation='relu',
                                               name='inception_4b_3x3_reduce')

        self.inception_4b_3x3 = conv_2d(self.inception_4b_3x3_reduce, 224, filter_size=3, activation='relu',
                                        name='inception_4b_3x3')

        self.inception_4b_5x5_reduce = conv_2d(self.inception_4a_output, 24, filter_size=1, activation='relu',
                                               name='inception_4b_5x5_reduce')

        self.inception_4b_5x5 = conv_2d(self.inception_4b_5x5_reduce, 64, filter_size=5, activation='relu',
                                        name='inception_4b_5x5')

        self.inception_4b_pool = max_pool_2d(self.inception_4a_output, kernel_size=3, strides=1,
                                             name='inception_4b_pool')

        self.inception_4b_pool_proj = conv_2d(self.inception_4b_pool, 64, filter_size=1, activation='relu',
                                              name='inception_4b_pool_proj')

        self.inception_4b_output = merge(
            [self.inception_4b_1x1, self.inception_4b_3x3, self.inception_4b_5x5, self.inception_4b_pool_proj],
            mode='concat', axis=3, name='inception_4b_output')

        self.inception_4c_1x1 = conv_2d(self.inception_4b_output, 128, filter_size=1, activation='relu',
                                        name='inception_4c_1x1')

        self.inception_4c_3x3_reduce = conv_2d(self.inception_4b_output, 128, filter_size=1, activation='relu',
                                               name='inception_4c_3x3_reduce')

        self.inception_4c_3x3 = conv_2d(self.inception_4c_3x3_reduce, 256, filter_size=3, activation='relu',
                                        name='inception_4c_3x3')

        self.inception_4c_5x5_reduce = conv_2d(self.inception_4b_output, 24, filter_size=1, activation='relu',
                                               name='inception_4c_5x5_reduce')

        self.inception_4c_5x5 = conv_2d(self.inception_4c_5x5_reduce, 64, filter_size=5, activation='relu',
                                        name='inception_4c_5x5')

        self.inception_4c_pool = max_pool_2d(self.inception_4b_output, kernel_size=3, strides=1,
                                             name='inception_4c_pool')

        self.inception_4c_pool_proj = conv_2d(self.inception_4c_pool, 64, filter_size=1, activation='relu',
                                              name='inception_4c_pool_proj')

        self.inception_4c_output = merge(
            [self.inception_4c_1x1, self.inception_4c_3x3, self.inception_4c_5x5, self.inception_4c_pool_proj],
            mode='concat', axis=3, name='inception_4c_output')

        self.inception_4d_1x1 = conv_2d(self.inception_4c_output, 112, filter_size=1, activation='relu',
                                        name='inception_4d_1x1')

        self.inception_4d_3x3_reduce = conv_2d(self.inception_4c_output, 144, filter_size=1, activation='relu',
                                               name='inception_4d_3x3_reduce')

        self.inception_4d_3x3 = conv_2d(self.inception_4d_3x3_reduce, 288, filter_size=3, activation='relu',
                                        name='inception_4d_3x3')

        self.inception_4d_5x5_reduce = conv_2d(self.inception_4c_output, 32, filter_size=1, activation='relu',
                                               name='inception_4d_5x5_reduce')

        self.inception_4d_5x5 = conv_2d(self.inception_4d_5x5_reduce, 64, filter_size=5, activation='relu',
                                        name='inception_4d_5x5')

        self.inception_4d_pool = max_pool_2d(self.inception_4c_output, kernel_size=3, strides=1,
                                             name='inception_4d_pool')

        self.inception_4d_pool_proj = conv_2d(self.inception_4d_pool, 64, filter_size=1, activation='relu',
                                              name='inception_4d_pool_proj')

        self.inception_4d_output = merge(
            [self.inception_4d_1x1, self.inception_4d_3x3, self.inception_4d_5x5, self.inception_4d_pool_proj],
            mode='concat', axis=3, name='inception_4d_output')

        self.inception_4e_1x1 = conv_2d(self.inception_4d_output, 256, filter_size=1, activation='relu',
                                        name='inception_4e_1x1')

        self.inception_4e_3x3_reduce = conv_2d(self.inception_4d_output, 160, filter_size=1, activation='relu',
                                               name='inception_4e_3x3_reduce')

        self.inception_4e_3x3 = conv_2d(self.inception_4e_3x3_reduce, 320, filter_size=3, activation='relu',
                                        name='inception_4e_3x3')

        self.inception_4e_5x5_reduce = conv_2d(self.inception_4d_output, 32, filter_size=1, activation='relu',
                                               name='inception_4e_5x5_reduce')

        self.inception_4e_5x5 = conv_2d(self.inception_4e_5x5_reduce, 128, filter_size=5, activation='relu',
                                        name='inception_4e_5x5')

        self.inception_4e_pool = max_pool_2d(self.inception_4d_output, kernel_size=3, strides=1,
                                             name='inception_4e_pool')

        self.inception_4e_pool_proj = conv_2d(self.inception_4e_pool, 128, filter_size=1, activation='relu',
                                              name='inception_4e_pool_proj')

        self.inception_4e_output = merge(
            [self.inception_4e_1x1, self.inception_4e_3x3, self.inception_4e_5x5, self.inception_4e_pool_proj],
            axis=3, mode='concat', name='inception_4e_output')

        self.pool4_3x3_s2 = max_pool_2d(self.inception_4e_output, kernel_size=3, strides=2, name='pool4_3x3_s2')

        self.inception_5a_1x1 = conv_2d(self.pool4_3x3_s2, 256, filter_size=1, activation='relu',
                                        name='inception_5a_1x1')

        self.inception_5a_3x3_reduce = conv_2d(self.pool4_3x3_s2, 160, filter_size=1, activation='relu',
                                               name='inception_5a_3x3_reduce')

        self.inception_5a_3x3 = conv_2d(self.inception_5a_3x3_reduce, 320, filter_size=3, activation='relu',
                                        name='inception_5a_3x3')

        self.inception_5a_5x5_reduce = conv_2d(self.pool4_3x3_s2, 32, filter_size=1, activation='relu',
                                               name='inception_5a_5x5_reduce')

        self.inception_5a_5x5 = conv_2d(self.inception_5a_5x5_reduce, 128, filter_size=5, activation='relu',
                                        name='inception_5a_5x5')

        self.inception_5a_pool = max_pool_2d(self.pool4_3x3_s2, kernel_size=3, strides=1, name='inception_5a_pool')

        self.inception_5a_pool_proj = conv_2d(self.inception_5a_pool, 128, filter_size=1, activation='relu',
                                              name='inception_5a_pool_proj')

        self.inception_5a_output = merge(
            [self.inception_5a_1x1, self.inception_5a_3x3, self.inception_5a_5x5, self.inception_5a_pool_proj],
            axis=3, mode='concat', name="inception_5a_output")

        self.inception_5b_1x1 = conv_2d(self.inception_5a_output, 384, filter_size=1, activation='relu',
                                        name='inception_5b_1x1')

        self.inception_5b_3x3_reduce = conv_2d(self.inception_5a_output, 192, filter_size=1, activation='relu',
                                               name='inception_5b_3x3_reduce')
        self.inception_5b_3x3 = conv_2d(self.inception_5b_3x3_reduce, 384, filter_size=3, activation='relu',
                                        name='inception_5b_3x3')

        self.inception_5b_5x5_reduce = conv_2d(self.inception_5a_output, 48, filter_size=1, activation='relu',
                                               name='inception_5b_5x5_reduce')

        self.inception_5b_5x5 = conv_2d(self.inception_5b_5x5_reduce, 128, filter_size=5, activation='relu',
                                        name='inception_5b_5x5')

        self.inception_5b_pool = max_pool_2d(self.inception_5a_output, kernel_size=3, strides=1,
                                             name='inception_5b_pool')

        self.inception_5b_pool_proj = conv_2d(self.inception_5b_pool, 128, filter_size=1, activation='relu',
                                              name='inception_5b_pool_proj')
        self.inception_5b_output = merge(
            [self.inception_5b_1x1, self.inception_5b_3x3, self.inception_5b_5x5, self.inception_5b_pool_proj],
            axis=3, mode='concat', name='inception_5b_output')

        self.pool5_7x7_s1 = avg_pool_2d(self.inception_5b_output, kernel_size=7, strides=1, name='pool5_7x7_s1')
        self.pool5_7x7_s1 = dropout(self.pool5_7x7_s1, 0.4, name="dropout_7x7_s1")

        # Output
        self.loss3_classifier = fully_connected(self.pool5_7x7_s1, label_size, activation='softmax',
                                                name='loss3_classifier')

        self.network = regression(self.loss3_classifier, optimizer='momentum',
                                  loss='categorical_crossentropy',
                                  learning_rate=0.001)

        self.model = tflearn.DNN(self.network, tensorboard_verbose=0,
                                 checkpoint_path='traffic-object-detection.tfl.ckpt')

    # Train
    def train(self, train_data, train_label, a_mount_of_steps, batch_size):
        self.model.fit(train_data, train_label, n_epoch=a_mount_of_steps, batch_size=batch_size,
                       snapshot_epoch=False, shuffle=True)

    # Save model
    def save(self, model_path_name):
        self.model.save(model_path_name)

    # Load data model
    def load(self, model_path_name):
        self.model.load(model_path_name)

    # Predict image
    # image_input can be one array or multi array
    def predict(self, image_input):
        return self.model.predict([image_input])
