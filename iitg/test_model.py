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
model = load_model(config.MODEL_PATH)

with open(config.MODEL_LABELS, "rb") as f:
    lb = pickle.load(f)

# load image from test path
for (i, path) in enumerate(paths.list_images(config.TEST_DATA_PATH)):
    print(f'[INFO] processing {i + 1}-th image')

    charList = seg_image(path)

    predictions = ''

    for char in charList:
        image = cv.cvtColor(char, cv.COLOR_BGR2GRAY)
        image = captchahelper.preprocess(image, config.INPUT_SIZE,config.INPUT_SIZE)
        image = img_to_array(image)
        data = np.expand_dims(image, axis=0) / 255.0
        pred = model.predict(data).argmax(axis=1)[0]
        predictions += lb.classes_[pred]

    image = plt.imread(path)
    plt.title(f'Prediction: {predictions}',fontsize=30)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image)
    plt.show()
