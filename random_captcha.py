import numpy as np
from imutils import paths
from shutil import copy

INPUT = 'iitg/datasets/testset'
TRAIN_OUTPUT = 'iitg/datasets/trainset'
TEST_OUTPUT = 'iitg/datasets/testset'
TEST_SIZE = 100

allPaths = list(paths.list_images(INPUT))

testSet = set(np.random.choice(allPaths, size=TEST_SIZE, replace=False))
# print(len(testSet))

for (i, imagePath) in enumerate(allPaths):
    if imagePath in testSet:
        copy(imagePath, TEST_OUTPUT)
    else:
        copy(imagePath, TRAIN_OUTPUT)

    print(f'[INFO] copy {i + 1}-th done...')
