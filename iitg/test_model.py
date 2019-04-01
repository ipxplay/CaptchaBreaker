import argparse
import os
import pickle

import cv2 as cv
import numpy as np
from imutils import paths
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from iitg import config
from iitg.preprocess.seg_image import seg_image
from iitg.train_model import read_data_labels
from pyimagesearch.utils import captchahelper


def init():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--no', required=True)
    return vars(ap.parse_args())


def load_model_labels(modelPath, lablesPath):
    print("[INFO] loading pre-trained network...")
    model = load_model(modelPath)

    with open(lablesPath, "rb") as f:
        lb = pickle.load(f)

    return model, lb


def test_model(no, modelPath, path):
    """test model from the whole picture"""

    model, lb = load_model_labels(modelPath, config.MODEL_LABELS)

    correct = 0
    amount = 0

    # load image from test path
    for (i, path) in enumerate(paths.list_images(path)):
        # print(f'[INFO] processing {i + 1}-th image')

        charList = seg_image(path)

        real = os.path.basename(path).split('.')[0]
        predictions = ''

        for char in charList:
            image = cv.cvtColor(char, cv.COLOR_BGR2GRAY)
            image = captchahelper.preprocess(image, config.INPUT_SIZE, config.INPUT_SIZE)
            image = img_to_array(image)
            data = np.expand_dims(image, axis=0) / 255.0
            pred = model.predict(data)
            pred = lb.inverse_transform(pred)[0]
            predictions += pred

        if predictions == real:
            correct += 1

        amount += 1

        print(f'{i + 1}-th: real is {real}, prediction is {predictions}, result is {predictions == real}')
        if amount % 100 == 0:
            print(f'[INFO] {amount} had done...')

        # image = plt.imread(path)
        # plt.title(f'Prediction: {predictions}', fontsize=30)
        # plt.xticks([]), plt.yticks([])
        # plt.imshow(image)
        # plt.show()

    print(f'the correct/amount is : {correct}/{amount}')
    print(f'the experiment no is {no}')

    return correct, amount


def test_model_2(no, modelPath, path):
    """test the testset"""

    model, lb = load_model_labels(modelPath, config.MODEL_LABELS)
    data, labels = read_data_labels(path)
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    preds = model.predict(data)
    preds = lb.inverse_transform(preds)
    amount = len(labels)
    assert len(preds) == len(labels)
    ac = 0
    for label, pred in zip(labels, preds):
        if label == pred:
            ac += 1

    print(f'[INFO] ac/amount: {ac}/{amount}')
    print(f'[INFO] the experiment no is: {no}')
    return ac, amount


if __name__ == '__main__':
    args = init()
    modelPath = os.path.sep.join([config.OUTPUT_PATH, f'{args["no"]}',
                                  f'{args["no"]}.hdf5'])
    test_model(args['no'], modelPath, 'datasets/allset')
