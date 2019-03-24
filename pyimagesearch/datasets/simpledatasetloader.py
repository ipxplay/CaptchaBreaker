import os

import cv2

import numpy as np


class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # the list of features and labels
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label
            # assuming that our path has follow format:
            # .../dateset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our preprocessed image as a "feature vector"
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f'[INFO] processed {i+1}/{len(imagePaths)}')

        # return a tuple of the data and labels
        return np.array(data), np.array(labels)
