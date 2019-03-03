from imutils import paths
import cv2 as cv
from keras.callbacks import ModelCheckpoint

from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.utils import captchahelper
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from pyimagesearch.nn.conv import LeNet

from iitg import config


def read_data_labels():
    data, labels = [], []

    for imagePath in paths.list_images(config.PREPROCESS_DATA_PATH):
        image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
        image = captchahelper.preprocess(image, 45, 45)
        # return a 3D Numpy array
        image = img_to_array(image)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    return data, labels


def prepare_data_labels(data, labels):
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    trainX, testX, trainY, testY = train_test_split(
        data, labels, test_size=0.25, random_state=42)

    lb = LabelBinarizer().fit(trainY)
    trainY = lb.transform(trainY)
    testY = lb.transform(testY)

    # save the encode label for decoding
    with open(config.MODEL_LABELS, "wb") as f:
        pickle.dump(lb, f)

    return trainX, testX, trainY, testY, lb


def train_model(trainX, testX, trainY, testY, lb):
    print('[INFO] compiling model...')
    model = LeNet.build(width=45, height=45, depth=1, classes=28)
    opt = SGD(lr=0.01, decay=0.01 / 10, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(config.MODEL_PATH, monitor='val_loss', mode='min',
                                 save_best_only=True, verbose=1)

    figPath = os.path.sep.join([config.OUTPUT_PATH, f'{os.getpid()}.png'])
    jsonPath = os.path.sep.join([config.OUTPUT_PATH, f'{os.getpid()}.json'])
    callbacks = [checkpoint, TrainingMonitor(figPath, jsonPath)]

    print('[INFO] training network...')
    model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=10, callbacks=callbacks, verbose=2)

    print('[INFO] evaluating network...')
    preds = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                preds.argmax(axis=1), target_names=lb.classes_))


if __name__ == '__main__':
    train_model(*prepare_data_labels(*read_data_labels()))
