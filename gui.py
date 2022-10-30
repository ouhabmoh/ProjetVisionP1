import sys

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLineEdit, QLabel, QHBoxLayout

import Yc


class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.window_width, self.window_height = 500, 700
        self.resize(self.window_width, self.window_height)
        self.setWindowTitle('Stagnographie')

        layout = QHBoxLayout()
        layout1 = QVBoxLayout()
        self.load_button = QPushButton('&Load Image')
        self.load_button.clicked.connect(self.load_img)
        layout1.addWidget(self.load_button)
        self.label = QLabel('Write your secret text')
        layout1.addWidget(self.label)
        self.textbox = QLineEdit(self)
        layout1.addWidget(self.textbox)
        self.code_button = QPushButton('code')
        self.code_button.clicked.connect(self.code)
        layout1.addWidget(self.code_button)

        layout2 = QVBoxLayout()
        self.load_button2 = QPushButton('&Load Image')
        self.load_button2.clicked.connect(self.load_img)
        layout2.addWidget(self.load_button2)

        self.decode_button = QPushButton('decode')
        self.decode_button.clicked.connect(self.decode)
        layout2.addWidget(self.decode_button)

        self.label2 = QLabel()
        layout2.addWidget(self.label2)

        layout.addLayout(layout1)
        layout.addLayout(layout2)
        self.setLayout(layout)

    def load_img(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '', "Image files (*.jpg *.gif *.png)")
        if filename is None:
            return
        self.img_name = filename
        img = cv2.imread(filename)
        self.img = np.uint16(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) * 256)

    def decode(self):
        cv2.imshow('code', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image_secret_decod = Yc.decode(self.img)
        cv2.imshow('decode', image_secret_decod)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.image = QImage(image_secret_decod.data, image_secret_decod.shape[1], image_secret_decod.shape[0],
                                  QImage.Format_RGB888).rgbSwapped()
        self.label2.setPixmap(QPixmap.fromImage(self.image))
        # # Optional, resize window to image size
        # self.resize(pixmap.width(), pixmap.height())
    def code(self):
        sec_img = Yc.secret_image(self.textbox.text())
        # cv2.imshow('decode', sec_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        Yc.code(self.img, sec_img)
        cv2.imshow('decode', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_code = np.uint8(cv2.cvtColor(self.img, cv2.COLOR_YCrCb2BGR)/256)
        cv2.imwrite('image coder.jpg', img_code)


if __name__ == '__main__':
    # don't auto scale when drag app to a different monitor.
    # QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 17px;
        }
    ''')

    myApp = MyApp()
    myApp.show()
    # myApp.showMaximized()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')
