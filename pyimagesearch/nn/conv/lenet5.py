from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential


class LeNet5:
    """The optimized LeNet-5"""

    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        act = 'relu'
        dropout = 0.3
        k_size1 = 5
        k_size2 = 5
        filters1 = 32
        filters2 = 64

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        # first set of CONV=>RELU=>POOL layers
        model.add(Conv2D(filters1, (k_size1, k_size1),
                         input_shape=inputShape, padding='same'))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(dropout))

        # second set of CONV=>RELU=>POOL layers
        model.add(Conv2D(filters2, (k_size2, k_size2), padding='same'))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(act))
        model.add(Dropout(dropout))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
