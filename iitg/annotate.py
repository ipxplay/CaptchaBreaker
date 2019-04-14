import os

import cv2 as cv
from imutils import paths

from iitg.common.segment import seg_image

# for filename count


INPUT = 'datasets/allset'
OUTPUT = 'datasets/segmented'
counts = {}
for i, filePath in enumerate(paths.list_images(INPUT)):
    print(f'[INFO] processing the {i + 1} image')

    charImgs, _ = seg_image(filePath)
    chars = os.path.basename(filePath).split('.')[0]

    for img, c in zip(charImgs, chars):
        dstPath = os.path.sep.join([OUTPUT, c])
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)

        count = counts.get(c, 1)
        p = os.path.sep.join([dstPath, f'{str(count).zfill(6)}.png'])
        cv.imwrite(p, img)
        counts[c] = count + 1
