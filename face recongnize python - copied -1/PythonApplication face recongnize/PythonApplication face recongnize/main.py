# -*- coding: GBK -*-
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import os
import time
import logging  # 用于日志记录
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

##目录
# 读取当前工作目录
current_dir = os.getcwd()  # 获得当前文件的文件夹路径
# print('当前工作目录:',current_dir)
# parent_dir0 = os.path.dirname(current_dir)
# print("当前目录的上级目录:", parent_dir0)
# parent_dir1 = os.path.dirname(parent_dir0)
# print('当前目录的上上级目录:',parent_dir1)#这就是项目目录了


# 构建分类器目录:"C:\Users\agaci\Desktop\face recongnize python - copied\haarcascade_frontalface_default.xml"
rel_path_CascadeClassifierdir = 'haarcascade_frontalface_default.xml'
dir_CascadeClassifier = os.path.join(current_dir, rel_path_CascadeClassifierdir)
# print('分类器的目录', dir_CascadeClassifier)

# 构建存储图像目录:"C:\Users\agaci\Desktop\face recongnize python - copied\person"
rel_path_person = 'person'
dir_person = os.path.join(current_dir, rel_path_person)

# 配置日志
logging.basicConfig(level=logging.INFO)


class FaceRecognitionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('人脸识别')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 600, 400)

        self.start_button = QPushButton('开始识别', self)
        self.start_button.setGeometry(50, 480, 150, 50)
        self.start_button.clicked.connect(self.startRecognition)

        self.stop_button = QPushButton('停止识别', self)
        self.stop_button.setGeometry(220, 480, 150, 50)
        self.stop_button.clicked.connect(self.stopRecognition)

        self.add_photo_button = QPushButton('拍摄添加照片', self)
        self.add_photo_button.setGeometry(390, 480, 150, 50)
        self.add_photo_button.clicked.connect(self.addPhoto)

        self.show_files_button = QPushButton('显示文件', self)
        self.show_files_button.setGeometry(560, 480, 150, 50)
        self.show_files_button.clicked.connect(self.showFiles)

    def startRecognition(self):
        try:
            # 加载 Haar Cascade 分类器
            face_cascade = cv2.CascadeClassifier(dir_CascadeClassifier)

            # 打开摄像头并捕获实时图像
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # 读取 person 文件夹中的图像和姓名
            person_images = []
            person_names = []
            for filename in os.listdir(dir_person):
                if filename.endswith('.jpg'):
                    # 使用 utf-8 编码打开文件
                    with open(os.path.join(dir_person, filename), 'rb') as f:
                        person_images.append(cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR))
                    person_names.append(os.path.splitext(filename)[0])

            while True:
                start_time = time.time()  # 记录开始时间
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("无法获取摄像头图像")
                    break

                # 转换图像格式以进行人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 使用 Haar Cascade 分类器检测人脸
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    found_person = False
                    for i in range(len(person_images)):
                        person_image = person_images[i]
                        person_name = person_names[i]
                        person_gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
                        match = cv2.matchTemplate(gray[y:y + h, x:x + w], person_gray, cv2.TM_CCOEFF_NORMED)
                        if match.max() > 0.7:
                            found_person = True
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            frame = self.cv2AddChineseText(frame, person_name, (x + (w / 2) - 10, y - 30), (0, 255, 255), 30)
                            break

                    if not found_person:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap)

                # 限制帧率，添加一点延迟
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time  # 计算本次处理时间
                if elapsed_time < 0.03:  # 目标帧率约为 30 帧每秒
                    time.sleep(0.03 - elapsed_time)

                QApplication.processEvents()

            self.cap.release()
        except Exception as e:
            logging.error(f"发生错误: {e}")

    def stopRecognition(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            QApplication.quit()  # 退出程序

    def addPhoto(self):
        if not hasattr(self, 'cap'):
            # 加载 Haar Cascade 分类器
            face_cascade = cv2.CascadeClassifier(dir_CascadeClassifier)

            # 打开摄像头并捕获实时图像
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            ret, frame = self.cap.read()
            if ret:
                # 实时显示拍摄的照片
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap)

                new_name, ok = QInputDialog.getText(self, '输入新名称', '请输入图片名称:')
                if ok and new_name:
                    timestamp = int(time.time())
                    filename = f"{new_name}_{timestamp}.jpg"
                    filepath = os.path.join(dir_person, filename)
                    cv2.imwrite(filepath, frame)
                    logging.info(f"成功拍摄并保存照片: {filename}")
            else:
                logging.error("无法获取拍摄的图像")
        except Exception as e:
            logging.error(f"拍摄和保存照片时发生错误: {e}")

    def showFiles(self):
        file_list = os.listdir(dir_person)
        dialog = QDialog(self)
        dialog.setWindowTitle('文件列表')
        layout = QVBoxLayout()
        for filename in file_list:
            if filename.endswith('.jpg'):
                button = QPushButton(filename, dialog)
                button.clicked.connect(lambda checked, fname=filename: self.handleFileAction(fname))
                layout.addWidget(button)
        dialog.setLayout(layout)
        dialog.exec_()

    def handleFileAction(self, filename):
        action = QMessageBox.question(self, '操作选择', '你想对文件 {} 进行什么操作？'.format(filename),
                                      QMessageBox.Rename | QMessageBox.Delete | QMessageBox.Cancel)
        if action == QMessageBox.Rename:
            new_name, ok = QInputDialog.getText(self, '重命名', '输入新名称:')
            if ok and new_name:
                new_path = os.path.join(dir_person, new_name + '.jpg')
                os.rename(os.path.join(dir_person, filename), new_path)
                logging.info(f'文件 {filename} 已重命名为 {new_name}.jpg')
        elif action == QMessageBox.Delete:
            os.remove(os.path.join(dir_person, filename))
            logging.info(f'文件 {filename} 已删除')

    def cv2AddChineseText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        if isinstance(img, np.ndarray):  # 判断是否 OpenCV 图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")

        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回 OpenCV 格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    app = QApplication([])
    window = FaceRecognitionUI()
    window.show()
    app.exec_()