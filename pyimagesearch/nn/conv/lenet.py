from keras.layers import BatchNormalization, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        # first set of CONV=>RELU=>POOL layers
        model.add(Conv2D(6, (5, 5), padding='same',
                         input_shape=inputShape))
        model.add(Activation('relu'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))

        # second set of CONV=>RELU=>POOL layers
        model.add(Conv2D(16, (5, 5), padding='same',
                         input_shape=inputShape))
        model.add(Activation('relu'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        # model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-06)
        # opt = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model