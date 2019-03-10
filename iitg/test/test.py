import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

gray = cv.imread('y7fg6.png', cv.IMREAD_GRAYSCALE)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
# medianBlur for removing salt & pepper noise
blurred = cv.medianBlur(thresh, 3)
# cv.imwrite('blurrd.png', blurred)

kernel = np.ones((2, 2), np.uint8)
image = cv.erode(blurred, kernel, iterations=1)
image = cv.dilate(image, kernel, iterations=1)
# cv.imwrite('result.png',image)


# for test how many white pixels per col
#
# colWhiteList=[]
# totalWhite=0
# for col in range(width):
#     wCount=0
#     for row in range(height):
#         if image[row][col]==255:
#             wCount+=1
#             totalWhite+=1
#
#     colWhiteList.append(wCount)
# plt.plot(range(width),colWhiteList)
# plt.show()

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

# display the result of k-means
# image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
# define colors to be used
# colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
#           [255, 0, 255]]
#
# for i, (row, col) in enumerate(features):
#     # labels[0]=[0]
#     image[row][col] = colors[labels[i][0]]
#
# for _, col in centers:
#     col = int(col)
#     for i in range(height):
#         image[i][col] = [255, 255, 255]
#
# plt.imshow(image)
# plt.xticks([]), plt.yticks([])
# plt.show()

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

chars = 'y7fg6'
for img, c in zip(charImgs, chars):
    if not os.path.exists(c):
        os.mkdir(c)

    p = os.path.sep.join([c, str(c) + '.png'])
    cv.imwrite(p, img)
