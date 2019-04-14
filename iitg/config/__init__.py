from os import path

TRAIN_DATA_PATH = path.sep.join(['..', 'datasets', 'trainset'])
DEV_DATA_PATH = path.sep.join(['..', 'datasets', 'devset'])
TEST_DATA_PATH = path.sep.join(['..', 'datasets', 'testset'])

OUTPUT_PATH = path.sep.join(['..', 'output'])
MODEL_LABELS = path.sep.join([OUTPUT_PATH, 'labels.mat'])

INPUT_SIZE = 48
