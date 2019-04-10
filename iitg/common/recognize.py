import cv2 as cv
import numpy as np
from keras.preprocessing.image import img_to_array

from iitg import config
from pyimagesearch.utils import captchahelper


def recognize_whole(model, lb, charList):
    """recognize from the charList"""
    predictions = ''
    for char in charList:
        image = cv.cvtColor(char, cv.COLOR_BGR2GRAY)
        image = captchahelper.preprocess(image, config.INPUT_SIZE, config.INPUT_SIZE)
        image = img_to_array(image)
        data = np.expand_dims(image, axis=0) / 255.0
        pred = model.predict(data)
        pred = lb.inverse_transform(pred)[0]
        predictions += pred

    return predictions
