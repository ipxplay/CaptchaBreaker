import pickle
from keras.models import load_model
from iitg import config
import os


def load_model_lables(no):
    modelPath = os.path.sep.join([config.OUTPUT_PATH, str(no), f'{no}.hdf5'])
    model = load_model(modelPath)

    with open(config.MODEL_LABELS, 'rb') as f:
        lb = pickle.load(f)

    return model, lb
