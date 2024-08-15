# -*- coding: GBK -*-
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import os
import time
import logging  # ������־��¼
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

##Ŀ¼
# ��ȡ��ǰ����Ŀ¼
current_dir = os.getcwd()  # ��õ�ǰ�ļ����ļ���·��
# print('��ǰ����Ŀ¼:',current_dir)
# parent_dir0 = os.path.dirname(current_dir)
# print("��ǰĿ¼���ϼ�Ŀ¼:", parent_dir0)
# parent_dir1 = os.path.dirname(parent_dir0)
# print('��ǰĿ¼�����ϼ�Ŀ¼:',parent_dir1)#�������ĿĿ¼��


# ����������Ŀ¼:"C:\Users\agaci\Desktop\face recongnize python - copied\haarcascade_frontalface_default.xml"
rel_path_CascadeClassifierdir = 'haarcascade_frontalface_default.xml'
dir_CascadeClassifier = os.path.join(current_dir, rel_path_CascadeClassifierdir)
# print('��������Ŀ¼', dir_CascadeClassifier)

# �����洢ͼ��Ŀ¼:"C:\Users\agaci\Desktop\face recongnize python - copied\person"
rel_path_person = 'person'
dir_person = os.path.join(current_dir, rel_path_person)

# ������־
logging.basicConfig(level=logging.INFO)


class FaceRecognitionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('����ʶ��')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 600, 400)

        self.start_button = QPushButton('��ʼʶ��', self)
        self.start_button.setGeometry(50, 480, 150, 50)
        self.start_button.clicked.connect(self.startRecognition)

        self.stop_button = QPushButton('ֹͣʶ��', self)
        self.stop_button.setGeometry(220, 480, 150, 50)
        self.stop_button.clicked.connect(self.stopRecognition)

        self.add_photo_button = QPushButton('���������Ƭ', self)
        self.add_photo_button.setGeometry(390, 480, 150, 50)
        self.add_photo_button.clicked.connect(self.addPhoto)

        self.show_files_button = QPushButton('��ʾ�ļ�', self)
        self.show_files_button.setGeometry(560, 480, 150, 50)
        self.show_files_button.clicked.connect(self.showFiles)

    def startRecognition(self):
        try:
            # ���� Haar Cascade ������
            face_cascade = cv2.CascadeClassifier(dir_CascadeClassifier)

            # ������ͷ������ʵʱͼ��
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # ��������ͷ�ֱ���
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # ��ȡ person �ļ����е�ͼ�������
            person_images = []
            person_names = []
            for filename in os.listdir(dir_person):
                if filename.endswith('.jpg'):
                    # ʹ�� utf-8 ������ļ�
                    with open(os.path.join(dir_person, filename), 'rb') as f:
                        person_images.append(cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR))
                    person_names.append(os.path.splitext(filename)[0])

            while True:
                start_time = time.time()  # ��¼��ʼʱ��
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("�޷���ȡ����ͷͼ��")
                    break

                # ת��ͼ���ʽ�Խ����������
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ʹ�� Haar Cascade �������������
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

                # ����֡�ʣ����һ���ӳ�
                end_time = time.time()  # ��¼����ʱ��
                elapsed_time = end_time - start_time  # ���㱾�δ���ʱ��
                if elapsed_time < 0.03:  # Ŀ��֡��ԼΪ 30 ֡ÿ��
                    time.sleep(0.03 - elapsed_time)

                QApplication.processEvents()

            self.cap.release()
        except Exception as e:
            logging.error(f"��������: {e}")

    def stopRecognition(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            QApplication.quit()  # �˳�����

    def addPhoto(self):
        if not hasattr(self, 'cap'):
            # ���� Haar Cascade ������
            face_cascade = cv2.CascadeClassifier(dir_CascadeClassifier)

            # ������ͷ������ʵʱͼ��
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # ��������ͷ�ֱ���
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            ret, frame = self.cap.read()
            if ret:
                # ʵʱ��ʾ�������Ƭ
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap)

                new_name, ok = QInputDialog.getText(self, '����������', '������ͼƬ����:')
                if ok and new_name:
                    timestamp = int(time.time())
                    filename = f"{new_name}_{timestamp}.jpg"
                    filepath = os.path.join(dir_person, filename)
                    cv2.imwrite(filepath, frame)
                    logging.info(f"�ɹ����㲢������Ƭ: {filename}")
            else:
                logging.error("�޷���ȡ�����ͼ��")
        except Exception as e:
            logging.error(f"����ͱ�����Ƭʱ��������: {e}")

    def showFiles(self):
        file_list = os.listdir(dir_person)
        dialog = QDialog(self)
        dialog.setWindowTitle('�ļ��б�')
        layout = QVBoxLayout()
        for filename in file_list:
            if filename.endswith('.jpg'):
                button = QPushButton(filename, dialog)
                button.clicked.connect(lambda checked, fname=filename: self.handleFileAction(fname))
                layout.addWidget(button)
        dialog.setLayout(layout)
        dialog.exec_()

    def handleFileAction(self, filename):
        action = QMessageBox.question(self, '����ѡ��', '������ļ� {} ����ʲô������'.format(filename),
                                      QMessageBox.Rename | QMessageBox.Delete | QMessageBox.Cancel)
        if action == QMessageBox.Rename:
            new_name, ok = QInputDialog.getText(self, '������', '����������:')
            if ok and new_name:
                new_path = os.path.join(dir_person, new_name + '.jpg')
                os.rename(os.path.join(dir_person, filename), new_path)
                logging.info(f'�ļ� {filename} ��������Ϊ {new_name}.jpg')
        elif action == QMessageBox.Delete:
            os.remove(os.path.join(dir_person, filename))
            logging.info(f'�ļ� {filename} ��ɾ��')

    def cv2AddChineseText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        if isinstance(img, np.ndarray):  # �ж��Ƿ� OpenCV ͼƬ����
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # ����һ�������ڸ���ͼ���ϻ�ͼ�Ķ���
        draw = ImageDraw.Draw(img)
        # ����ĸ�ʽ
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")

        # �����ı�
        draw.text(position, text, textColor, font=fontStyle)
        # ת���� OpenCV ��ʽ
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    app = QApplication([])
    window = FaceRecognitionUI()
    window.show()
    app.exec_()