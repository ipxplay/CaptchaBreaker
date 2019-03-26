import numpy as np
from imutils import paths
from shutil import copy

INPUT = 'iitg/datasets/allset'
TRAIN_OUTPUT = 'iitg/datasets/trainset'
TEST_OUTPUT = 'iitg/datasets/testset'
# 3015*0.2
TEST_SIZE = 603

allPaths = list(paths.list_images(INPUT))

testSet = set(np.random.choice(allPaths, size=TEST_SIZE, replace=False))
assert len(testSet) == TEST_SIZE

for (i, imagePath) in enumerate(allPaths):
    if imagePath in testSet:
        copy(imagePath, TEST_OUTPUT)
    else:
        copy(imagePath, TRAIN_OUTPUT)

    print(f'[INFO] copy {i + 1}-th done...')
