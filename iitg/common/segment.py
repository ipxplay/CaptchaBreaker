import time

import cv2 as cv
import numpy as np
from keras.preprocessing.image import img_to_array

from iitg import config
from pyimagesearch.utils import captchahelper


def clock(func):
    """ a simple time decorator"""

    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        duration = time.perf_counter() - t0
        return result, duration

    return clocked


@clock
def gray_image(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


@clock
def binary_image(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]


@clock
def denoise_image(image):
    return cv.medianBlur(image, 3)


@clock
def erose_dialte_image(image):
    kernel = np.ones((2, 2), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    return cv.dilate(image, kernel, iterations=1)


@clock
def seg_from_image(image):
    height, width = image.shape

    # perpare data for clustering
    features = [[row, col]
                for row in range(height)
                for col in range(width)
                if image[row][col] == 255
                ]
    z = np.float32(features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    compactness, labels, centers = cv.kmeans(z, 5, None, criteria, 10, flags)

    # convert to color image for extracting
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # extract character
    # 45 is measured from the original figure
    charHeight, charWidth = height, 45
    charImgs = []
    centers = sorted(centers, key=lambda x: x[1])

    step = charWidth // 2

    for _, colNum in centers:
        colNum = int(colNum)
        left = (colNum - step) if (colNum - step) > 0 else 0
        right = (colNum + step) if (colNum + step) < width else width
        charImgs.append(image[0:charHeight, left:right])

    return charImgs


@clock
def seg_image(path):
    """Give a image, return the segmented chars list from the image"""
    image = cv.imread(path)
    image, _ = gray_image(image)
    image, _ = binary_image(image)
    image, _ = denoise_image(image)
    image, _ = erose_dialte_image(image)
    result, _ = seg_from_image(image)
    return result


@clock
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


def seg_image_2(path):
    """Give a path of image, return the segmented chars list from the image"""

    gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # medianBlur for removing salt & pepper noise
    blurred = cv.medianBlur(thresh, 3)

    kernel = np.ones((2, 2), np.uint8)
    image = cv.erode(blurred, kernel, iterations=1)
    image = cv.dilate(image, kernel, iterations=1)

    height, width = image.shape

    # perpare data for clustering
    features = [[row, col]
                for row in range(height)
                for col in range(width)
                if image[row][col] == 255
                ]
    z = np.float32(features)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    compactness, labels, centers = cv.kmeans(z, 5, None, criteria, 10, flags)

    # convert to color image for extracting
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # extract character
    # 45 is measured from the original figure
    charHeight, charWidth = height, 45
    charImgs = []
    centers = sorted(centers, key=lambda x: x[1])

    step = charWidth // 2

    for _, colNum in centers:
        colNum = int(colNum)
        left = (colNum - step) if (colNum - step) > 0 else 0
        right = (colNum + step) if (colNum + step) < width else width
        charImgs.append(image[0:charHeight, left:right])

    return charImgs
