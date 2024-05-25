# coding:utf-8

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QCloseEvent, QKeyEvent, QMoveEvent, QPixmap, QPainter, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
import ctypes, sys
import cv2
import numpy as np
from threading import Thread

class StepWindow(QWidget):
    def __init__(self, width, height, auto_start = False):
        super().__init__()
        self.init_ui(width, height, auto_start)

    def init_ui(self,width, height, auto_start = False):
        self.setGeometry(300,500,width,height)

        self.setWindowOpacity(0.5)

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("解答步骤执行窗格")
        self.navi = NaviWidget(self)
        # self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput)

        #创建图像显示部分
        layout = QVBoxLayout()
        self.img_label = QLabel(self)
        layout.addWidget(self.img_label)

        self.setLayout(layout)
        self.should_start = auto_start
    
    def show(self) -> None:
        self.navi.show()
        return super().show()
    
    def display_img(self, cv_img:np.ndarray):
        img_cp = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_cp.shape
        bytes_per_line = channel * width

        qImg = QImage(img_cp.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qImg))

class NaviWidget(QWidget):
    def __init__(self,binded_window:StepWindow):
        super().__init__()
        self.binded_window = binded_window
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setFixedSize(self.binded_window.size().width(), 10)
        self.binded_window.move(self.pos().x(), self.pos().y() + self.size().height())
        self.setWindowTitle("导航窗格")
        

    def moveEvent(self, event: QMoveEvent) -> None:
        self.binded_window.move(self.pos().x(), self.pos().y() + self.size().height() + 40)
        return super().moveEvent(event)
    
    # 没有设定自动开始时，焦点在导航上按Enter启动
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        print(f"Key {key} Pressed, QtEnter:{Qt.Key_Enter}, should_start={self.binded_window.should_start}")
        if key in [Qt.Key_Enter, Qt.Key_Return] and self.binded_window.should_start == False:
            print(f"Key {key} Pressed")
            self.binded_window.should_start = True
        
        return super().keyReleaseEvent(event)
    
    def closeEvent(self, a0: QCloseEvent) -> None:
        print("on close")
        self.binded_window.close()
        return super().closeEvent(a0)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = StepWindow(400,500)
    win.show()

    qt_thread = Thread(target=app.exec_)
    qt_thread.daemon = True
    qt_thread.start()

    input("Pause")

    img = cv2.imread("temp\clipboardimage_1716483591.png")
    win.display_img(img)

    input("Press Enter To Exit")