import argparse
import os

from imutils import paths

from iitg.common.loads import load_model_lables
from iitg.common.segment import seg_image, recognize_whole


def init():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--no', required=True)
    return vars(ap.parse_args())


def test_model(no, path):
    """test model from the whole picture"""
    model, lb = load_model_lables(no)

    correct = 0
    amount = 0
    allTime = 0

    # load image from test path
    for (i, path) in enumerate(paths.list_images(path)):

        charList, time = seg_image(path)
        allTime += time
        predictions, time = recognize_whole(model, lb, charList)
        allTime += time

        real = os.path.basename(path).split('.')[0]
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

    return correct, amount, allTime


if __name__ == '__main__':
    args = init()
    no = args['no']
    correct, amount, allTime = test_model(no, '../datasets/allset')
    print(f'the correct/amount is : {correct}/{amount}')
    print(f'the avg time is {allTime / amount:.3}s')
    print(f'the experiment no is {no}')
