import argparse
import os
import pickle

import cv2 as cv
import numpy as np
from imutils import paths
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from iitg import config
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.nn.conv.lenet5 import LeNet5
from pyimagesearch.utils import captchahelper


def init():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--no', required=True)
    args = vars(ap.parse_args())
    print(f'the exp no is {args["no"]}')
    return args


def read_data_labels(path):
    data, labels = [], []
    for imagePath in paths.list_images(path):
        image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
        image = captchahelper.preprocess(image, config.INPUT_SIZE, config.INPUT_SIZE)
        # return a 3D Numpy array
        image = img_to_array(image)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    return data, labels


def prepare_data_labels(data, labels):
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    if os.path.exists(config.MODEL_LABELS):
        with open(config.MODEL_LABELS, 'rb') as f:
            lb = pickle.load(f)
    else:
        lb = LabelBinarizer().fit(labels)
        # save the encode label for decoding
        with open(config.MODEL_LABELS, "wb") as f:
            pickle.dump(lb, f)

    labels = lb.transform(labels)
    return data, labels


def train_model(trainX, trainY, devX, devY, args):
    PATH = os.path.sep.join([config.OUTPUT_PATH, args['no']])
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    modelPath = os.path.sep.join([PATH, f'{args["no"]}.hdf5'])
    checkpoint = ModelCheckpoint(modelPath, monitor='val_loss', mode='min',
                                 save_best_only=True, verbose=1)

    figPath = os.path.sep.join([PATH, f'{args["no"]}.png'])
    jsonPath = os.path.sep.join([PATH, f'{args["no"]}.json'])
    callbacks = [checkpoint, TrainingMonitor(figPath, jsonPath)]

    aug = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    print('[INFO] training network...')
    if  os.path.exists(modelPath):
        model = load_model(modelPath)
        print('[INFO] loaded checkpoint...')
    else:
        print('[INFO] compiling model...')
        model = LeNet5.build(width=config.INPUT_SIZE, height=config.INPUT_SIZE, depth=1, classes=28)
        opt = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    batchSize = 128
    # model.fit(trainX, trainY, validation_data=(devX, devY), batch_size=batchSize,
    #           epochs=20, verbose=2, callbacks=callbacks)

    # all random state must set random seed
    model.fit_generator(
        aug.flow(trainX, trainY, batch_size=batchSize, seed=42),
        validation_data=(devX, devY),
        steps_per_epoch=len(trainX) // batchSize,
        epochs=20, callbacks=callbacks, verbose=2)

    model = load_model(modelPath)
    return model, batchSize


def evaluate_model(model, lb, batch_size, devX, devY):
    print('[INFO] evaluating network...')
    preds = model.predict(devX, batch_size=batch_size)
    print(classification_report(devY.argmax(axis=1),
                                preds.argmax(axis=1), target_names=lb.classes_))


def test_model(model, lb, path):
    """test the testset"""
    data, labels = read_data_labels(path)
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    preds = model.predict(data)
    preds = lb.inverse_transform(preds)
    amount = len(labels)
    assert len(preds) == len(labels)
    ac = sum(labels == preds)

    return ac, amount


if __name__ == '__main__':
    args = init()
    trainX, trainY = prepare_data_labels(*read_data_labels(config.TRAIN_DATA_PATH))
    devX, devY = prepare_data_labels(*read_data_labels(config.DEV_DATA_PATH))
    with open(config.MODEL_LABELS, 'rb') as f:
        lb = pickle.load(f)
    model, batch_size = train_model(trainX, trainY, devX, devY, args)
    evaluate_model(model, lb, batch_size, devX, devY)
    testAcc, testAll = test_model(model, lb, config.TEST_DATA_PATH)
    print(f'test: accuary/amount is {testAcc}/{testAll}')
    print(f'[INFO] the experiment no is {args["no"]}')
