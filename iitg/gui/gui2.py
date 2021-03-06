# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui2.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(570, 489)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(50, 20, 50, 20)
        self.verticalLayout.setSpacing(20)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(0, 100))
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.little1 = QtWidgets.QLabel(self.centralwidget)
        self.little1.setObjectName("little1")
        self.horizontalLayout.addWidget(self.little1)
        self.little2 = QtWidgets.QLabel(self.centralwidget)
        self.little2.setObjectName("little2")
        self.horizontalLayout.addWidget(self.little2)
        self.little3 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.little3.sizePolicy().hasHeightForWidth())
        self.little3.setSizePolicy(sizePolicy)
        self.little3.setMinimumSize(QtCore.QSize(0, 100))
        self.little3.setObjectName("little3")
        self.horizontalLayout.addWidget(self.little3)
        self.little4 = QtWidgets.QLabel(self.centralwidget)
        self.little4.setObjectName("little4")
        self.horizontalLayout.addWidget(self.little4)
        self.little5 = QtWidgets.QLabel(self.centralwidget)
        self.little5.setObjectName("little5")
        self.horizontalLayout.addWidget(self.little5)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.result = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.result.sizePolicy().hasHeightForWidth())
        self.result.setSizePolicy(sizePolicy)
        self.result.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.result.setFont(font)
        self.result.setObjectName("result")
        self.verticalLayout.addWidget(self.result)
        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, -1, 20, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setAutoFillBackground(False)
        self.pushButton_1.setObjectName("pushButton_1")
        self.verticalLayout_2.addWidget(self.pushButton_1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setAutoFillBackground(False)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setAutoFillBackground(False)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setAutoFillBackground(False)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_2.addWidget(self.pushButton_4)
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setAutoFillBackground(False)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_2.addWidget(self.pushButton_5)
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setAutoFillBackground(False)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_2.addWidget(self.pushButton_6)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.pushButton_1.clicked.connect(MainWindow.browseFile)
        self.pushButton_2.clicked.connect(MainWindow.grayImage)
        self.pushButton_3.clicked.connect(MainWindow.binaryImage)
        self.pushButton_4.clicked.connect(MainWindow.denoiseImage)
        self.pushButton_5.clicked.connect(MainWindow.eroseDilateImage)
        self.pushButton_6.clicked.connect(MainWindow.cutChars)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "验证码识别模拟"))
        self.label.setText(_translate("MainWindow", "                 Picture"))
        self.little1.setText(_translate("MainWindow", "little1"))
        self.little2.setText(_translate("MainWindow", "little2"))
        self.little3.setText(_translate("MainWindow", "little3"))
        self.little4.setText(_translate("MainWindow", "little4"))
        self.little5.setText(_translate("MainWindow", "little5"))
        self.pushButton_1.setText(_translate("MainWindow", "选择文件"))
        self.pushButton_2.setText(_translate("MainWindow", "灰度化"))
        self.pushButton_3.setText(_translate("MainWindow", "二值化"))
        self.pushButton_4.setText(_translate("MainWindow", "去噪"))
        self.pushButton_5.setText(_translate("MainWindow", "腐蚀和膨胀"))
        self.pushButton_6.setText(_translate("MainWindow", "分割识别"))


