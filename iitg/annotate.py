import os
from imutils import paths
import cv2 as cv
import numpy as np

from iitg import config

# for filename count
counts = {}

for i, filePath in enumerate(paths.list_images(config.TRAIN_DATA_PATH)):
    print(f'[INFO] processing the {i + 1} image')

    gray = cv.imread(filePath, cv.IMREAD_GRAYSCALE)
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
    charHeight, charWeight = height, 45
    charImgs = []
    centers = sorted(centers, key=lambda x: x[1])

    step = charWeight // 2

    for _, colNum in centers:
        colNum = int(colNum)
        left = (colNum - step) if (colNum - step) > 0 else 0
        right = (colNum + step) if (colNum + step) < width else width
        charImgs.append(image[0:charHeight, left:right])

    chars = os.path.basename(filePath).split('.')[0]

    for img, c in zip(charImgs, chars):
        dstPath = os.path.sep.join([config.PREPROCESS_DATA_PATH, c])
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)

        count = counts.get(c, 1)
        p = os.path.sep.join([dstPath, f'{str(count).zfill(6)}.png'])
        cv.imwrite(p, img)
        counts[c] = count + 1
