import numpy as np
from imutils import paths
from shutil import copy
import os

INPUT = 'datasets/segmented'
TRAIN_OUTPUT = 'datasets/trainset'
DEV_OUTPUT = 'datasets/devset'
TEST_OUTPUT = 'datasets/testset'


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


for label in os.listdir(INPUT):
    labelPath = os.path.sep.join([INPUT, label])
    trainDst = os.path.sep.join([TRAIN_OUTPUT, label])
    devDst = os.path.sep.join([DEV_OUTPUT, label])
    testDst = os.path.sep.join([TEST_OUTPUT, label])

    create_dirs(trainDst)
    create_dirs(devDst)
    create_dirs(testDst)

    allList = list(paths.list_images(labelPath))
    size = len(allList)
    devSize = int(size * 0.2)
    testSize = devSize

    np.random.seed(42)
    devList = np.random.choice(allList, size=devSize, replace=False)
    print(f'[INFO] dev size: {len(devList)}')
    leftList = list(set(allList) - set(devList))
    testList = np.random.choice(leftList, size=testSize, replace=False)
    print(f'[INFO] test size: {len(testList)}')

    for (i, imagePath) in enumerate(allList):
        if imagePath in devList:
            copy(imagePath, devDst)
        elif imagePath in testList:
            copy(imagePath, testDst)
        else:
            copy(imagePath, trainDst)

        print(f'[INFO] copy {i + 1}-th done...')