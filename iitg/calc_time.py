import argparse
import os
import pickle
from time import time

from imutils import paths
from keras.models import load_model

from iitg import config
from iitg.common.recognize import recognize_whole
from iitg.common.seg_image import seg_image


def init():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--no', required=True)
    return vars(ap.parse_args())


def calc_time(no, modelPath, path):
    """test model from the whole picture"""
    model = load_model(modelPath)

    with open(config.MODEL_LABELS, "rb") as f:
        lb = pickle.load(f)

    amount = 0
    start = time()
    # load image from test path
    for (i, path) in enumerate(paths.list_images(path)):
        charList = seg_image(path)
        recognize_whole(model, lb, charList)
        amount += 1

    duration = time() - start
    print(f'avg time is {duration / amount}')
    print(f'the experiment no is {no}')


if __name__ == '__main__':
    args = init()
    modelPath = os.path.sep.join([config.OUTPUT_PATH, f'{args["no"]}',
                                  f'{args["no"]}.hdf5'])
    calc_time(args['no'], modelPath, 'datasets/allset')
