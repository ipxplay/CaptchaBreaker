import os
import sys

import cv2 as cv
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from iitg.common import segment
from iitg.common.loads import load_model_lables
from iitg.gui.gui2 import Ui_MainWindow


def toQImage(image):
    # print(type(image))
    if len(image.shape) == 3:
        qFormat = QImage.Format_RGB888
    else:
        qFormat = QImage.Format_Grayscale8
    qImage = QImage(image.data,
                    image.shape[1],
                    image.shape[0],
                    image.strides[0],
                    qFormat)
    qImage = qImage.rgbSwapped()
    return QPixmap.fromImage(qImage)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

    def browseFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()", "../datasets/allset", "All Files (*)",
            options=options)

        # orignal captcha chars
        self.oringal = os.path.basename(fileName).split('.')[0]
        # count time
        self.time = 0

        self.image = cv.imread(fileName)
        self.label.setPixmap(toQImage(self.image))

    def grayImage(self):
        self.image, time = segment.gray_image(self.image)
        self.time += time
        self.label.setPixmap(toQImage(self.image))

    def binaryImage(self):
        self.image, time = segment.binary_image(self.image)
        self.time += time
        self.label.setPixmap(toQImage(self.image))

    def denoiseImage(self):
        self.image, time = segment.denoise_image(self.image)
        self.time += time
        self.label.setPixmap(toQImage(self.image))

    def eroseDilateImage(self):
        self.image, time = segment.erose_dialte_image(self.image)
        self.time += time
        self.label.setPixmap(toQImage(self.image))

    def cutChars(self):
        charImgs, time = segment.seg_from_image(self.image)
        self.time += time

        littles = [self.little1, self.little2, self.little3,
                   self.little4, self.little5]

        for charImg, little in zip(charImgs, littles):
            # print(type(charImg))
            little.setPixmap(toQImage(np.array(charImg)))

        model, lb = load_model_lables('3.3')

        result, time = segment.recognize_whole(model, lb, charImgs)
        self.time += time

        text1 = f'原始验证码：{self.oringal}'
        text2 = f'识别验证码：{result}'

        text3 = f'识别结果：{self.oringal == result}'
        text4 = f'识别用时：{self.time:.3}s'
        self.result.setText(text1)
        self.result.append(text2)
        self.result.append("")
        self.result.append(text3)
        self.result.append(text4)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
