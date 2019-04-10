import argparse
import os
import pickle

from imutils import paths
from keras.models import load_model

from iitg import config
from iitg.common.recognize import recognize_whole
from iitg.common.seg_image import seg_image


def init():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--no', required=True)
    return vars(ap.parse_args())


def test_model(no, modelPath, path):
    """test model from the whole picture"""
    model = load_model(modelPath)

    with open(config.MODEL_LABELS, "rb") as f:
        lb = pickle.load(f)

    correct = 0
    amount = 0

    # load image from test path
    for (i, path) in enumerate(paths.list_images(path)):
        # print(f'[INFO] processing {i + 1}-th image')

        charList = seg_image(path)

        real = os.path.basename(path).split('.')[0]
        predictions = recognize_whole(model, lb, charList)

        if predictions == real:
            correct += 1

        amount += 1

        print(f'{i + 1}-th: real is {real}, prediction is {predictions}, result is {predictions == real}')
        if amount % 100 == 0:
            print(f'[INFO] {amount} had done...')

        # image = plt.imread(path)
        # plt.title(f'Prediction: {predictions}', fontsize=30)
        # plt.xticks([]), plt.yticks([])
        # plt.imshow(image)
        # plt.show()

    print(f'the correct/amount is : {correct}/{amount}')
    print(f'the experiment no is {no}')

    return correct, amount


if __name__ == '__main__':
    args = init()
    modelPath = os.path.sep.join([config.OUTPUT_PATH, f'{args["no"]}',
                                  f'{args["no"]}.hdf5'])
    test_model(args['no'], modelPath, 'datasets/allset')
