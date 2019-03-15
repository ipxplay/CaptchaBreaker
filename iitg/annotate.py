import os
from imutils import paths
import cv2 as cv
from iitg.preprocess.seg_image import seg_image

from iitg import config

# for filename count
counts = {}

for i, filePath in enumerate(paths.list_images(config.TRAIN_DATA_PATH)):
    print(f'[INFO] processing the {i + 1} image')

    charImgs = seg_image(filePath)
    chars = os.path.basename(filePath).split('.')[0]

    for img, c in zip(charImgs, chars):
        dstPath = os.path.sep.join([config.PREPROCESS_DATA_PATH, c])
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)

        count = counts.get(c, 1)
        p = os.path.sep.join([dstPath, f'{str(count).zfill(6)}.png'])
        cv.imwrite(p, img)
        counts[c] = count + 1
