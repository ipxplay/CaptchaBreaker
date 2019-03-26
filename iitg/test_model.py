import argparse
import os
import pickle

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from pyimagesearch.utils import captchahelper

from iitg import config
from iitg.preprocess.seg_image import seg_image
from imutils import paths

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

print("[INFO] loading pre-trained network...")
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--no', required=True)
args = vars(ap.parse_args())
modelPath = os.path.sep.join([config.OUTPUT_PATH, f'{args["no"]}.hdf5'])
model = load_model(modelPath)

with open(config.MODEL_LABELS, "rb") as f:
    lb = pickle.load(f)

correct = 0
allCount = 0

# load image from test path
for (i, path) in enumerate(paths.list_images(config.TEST_DATA_PATH)):
    # print(f'[INFO] processing {i + 1}-th image')

    charList = seg_image(path)

    real = os.path.basename(path).split('.')[0]
    predictions = ''

    for char in charList:
        image = cv.cvtColor(char, cv.COLOR_BGR2GRAY)
        image = captchahelper.preprocess(image, config.INPUT_SIZE, config.INPUT_SIZE)
        image = img_to_array(image)
        data = np.expand_dims(image, axis=0) / 255.0
        pred = model.predict(data).argmax(axis=1)[0]
        predictions += lb.classes_[pred]

    if predictions == real:
        correct += 1

    allCount += 1

    print(f'{i + 1}-th: real is {real}, prediction is {predictions}, result is {predictions == real}')

    # image = plt.imread(path)
    # plt.title(f'Prediction: {predictions}', fontsize=30)
    # plt.xticks([]), plt.yticks([])
    # plt.imshow(image)
    # plt.show()

print(f'the correct rate is : {correct / allCount:.2f}')
