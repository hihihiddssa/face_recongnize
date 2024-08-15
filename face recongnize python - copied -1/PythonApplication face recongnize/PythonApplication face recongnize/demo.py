#-*-coding:GBK -*-
import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk,ImageDraw
import numpy as np
 
from PIL import ImageFont
 
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # �ж��Ƿ�OpenCVͼƬ����
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ����һ�������ڸ���ͼ���ϻ�ͼ�Ķ���
    draw = ImageDraw.Draw(img)
    # ����ĸ�ʽ
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    
    # �����ı�
    draw.text(position, text, textColor, font=fontStyle)
    # ת����OpenCV��ʽ
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
 
##�����Զ�������
font = ImageFont.truetype(r"C:\Users\agaci\Desktop\YaHei.Consolas.1.11b.ttf", size=30)
 
# ����Haar Cascade������
face_cascade = cv2.CascadeClassifier(r"C:\Users\agaci\Desktop\face recongnize python\haarcascade_frontalface_default.xml")
 
# ����GUI����
root = tk.Tk()#Tk �������Ӧ�ó���������ڡ�ͨ������ Tk ���ʵ�������丳ֵ�� root ���������Ǿͽ�����Ӧ�ó�����������ܡ�
root.geometry('640x480')#���ô��ڴ�С
root.title('face recongnize')#���ô��ڱ���
'''
        root = tk.Tk() ���д��봴����һ���������ڶ��� root ��
        root.geometry('640x480') �������ô��ڵĳ�ʼ��СΪ 640 ���ؿ�� 480 ���ظߡ�
        root.title('����ʶ��') ��Ϊ������������˱���Ϊ������ʶ��
'''
# ������ǩ������ʾͼ��
image_label = tk.Label(root)
image_label.pack()
'''
tk.Label(root)��Label �� tkinter ���е�һ���ؼ��࣬���ڴ�����ǩ��
root ��Ϊ�������ݸ� Label �Ĺ��캯������ʾ�����ǩ�������� root ����������С�

pack() ��һ�ּ��ι�����������
�����������Զ����ؼ������ڴ����У������ݴ��ڵĿ��ÿռ�������ѷ��õĿؼ���ȷ���ؼ���λ�úʹ�С��

'''
# ������ͷ������ʵʱͼ��
cap = cv2.VideoCapture(0)
'''
VideoCapture �����ڴ�����ͷ����Ƶ�ļ���������ƵԴ�л�ȡ��Ƶ֡��
����������У����� 0 ��ʾʹ��Ĭ�ϵ�����ͷ��ͨ���Ǽ�������õ�����ͷ��

���Ҫ����Ƶ�ļ��ж�ȡ��Ƶ��
���Խ��ļ�·����Ϊ�������ݣ����� cap = cv2.VideoCapture('video.mp4')
'''
# ���� PhotoImage ����
photo = None
 
# ��ȡperson�ļ����е�ͼ�������
person_images = []
person_names = []
for filename in os.listdir(r"C:\Users\agaci\Desktop\face recongnize python\person"):
    if filename.endswith('.jpg'):
        # ʹ��utf-8������ļ�
        with open(os.path.join(r"C:\Users\agaci\Desktop\face recongnize python\person", filename), 'rb') as f:
            person_images.append(cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR))
        person_names.append(os.path.splitext(filename)[0])
 
 
# ѭ����������ͷ�����ͼ��
while True:
    ret, frame = cap.read()
    if not ret:# ret ͨ����һ������ֵ��True �� False��
        break
 
    # ת��ͼ���ʽ�Խ����������
    # ʹ�� OpenCV ���е� cv2.cvtColor �����������ͼ�� frame �� BGR ��ɫ�ռ�ת��Ϊ�Ҷȣ�GRAY����ɫ�ռ䣬��������洢�ڱ��� gray ��
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # ʹ��Haar Cascade�������������
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    '''
    ʹ�� OpenCV �е� face_cascade ����� detectMultiScale �����ڻҶ�ͼ�� gray �м������
    
    detectMultiScale �����᷵��һ��������⵽����������ľ���������б�
    ���磬�����ͼ���гɹ���⵽������������faces ���ܻ������� [(x1, y1, w1, h1), (x2, y2, w2, h2)] ����ʽ������ (x, y) �Ǿ������Ͻǵ����꣬(w, h) �ֱ��Ǿ��εĿ�Ⱥ͸߶ȡ�
    
    scaleFactor=1.1 ��ʾ��ͼ���������ÿ������ͼ��ı������������˼�ⴰ���ڲ�ͬ�߶��ϵ����ų̶ȣ�ֵԽ�󣬼���ٶ�Խ�죬�����ܻ�©��һЩ��С��������ֵԽС��������ϸ���������������ӡ�
    minNeighbors=5 ��ʾÿ����ѡ��������Ӧ�ñ��������ھ��ε�����������������ڿ��Ƽ�⵽��������׼ȷ�ԡ�ֵԽ�������Խ�ͣ������ܻ�©��һЩ���Ѽ���������ֵԽС�����ܻ��⵽��������������
    
    '''
    # ��ͼ���п����⵽������
    for (x, y, w, h) in faces:
        # ��������Ƿ�����person�ļ����е�ĳ����
        found_person = False
        for i in range(len(person_images)):
            person_image = person_images[i]
            person_name = person_names[i]
            # ��personͼ��ת��Ϊ�Ҷ�ͼ���Խ��бȽ�
            person_gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
            # ����Ƿ������personͼ����ƥ�������
            match = cv2.matchTemplate(gray[y:y + h, x:x + w], person_gray, cv2.TM_CCOEFF_NORMED)
            '''
                gray[y:y + h, x:x + w]�����ǴӻҶ�ͼ�� gray ����ȡ�ĵ�ǰ��⵽����������
                person_gray�����Ǵ� person �ļ����л�ȡ��ĳ���˵ĻҶ�ͼ�������뵱ǰ��⵽�������������ƥ��Ƚ�
                cv2.TM_CCOEFF_NORMED������ָ����ƥ�䷽��������������£�ʹ�õ��ǡ���һ�������ϵ��������.���ַ������������ͼ������֮�������ԣ����������һ���� [-1, 1] �ķ�Χ�ڡ�ֵԽ�ӽ� 1 ��ʾƥ���Խ�ߣ�ֵԽ�ӽ� -1 ��ʾƥ���Խ��
            '''   
            if match.max() > 0.8:
                print(person_name)
                found_person = True
                # ��ͼ���п����������ʾ����
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                '''
                frame��Ҫ���ƾ��ε�ͼ��
                (x, y)�����ε����ϽǶ������ꡣ
                (x + w, y + h)�����ε����½Ƕ������ꡣͨ�����Ͻ����� (x, y) �Լ������Ŀ�� w �͸߶� h ��ȷ�����½ǵ�λ�á�
                (0, 255, 255)�����α߿����ɫ���� OpenCV �У���ɫͨ���� BGR ˳���ʾ������� (0, 255, 255) ��ʾ��ɫ����ɫͨ��ֵΪ 0 ����ɫͨ��ֵΪ 255 ����ɫͨ��ֵΪ 255 ����
                '''
                # ��ͼ���п����������ʾ����
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                frame = cv2AddChineseText(frame, person_name, (x + (w/2)-10, y - 30), (0, 255, 255), 30)
                break
 
        # ���û���ҵ�ƥ�������������ͼ���п������������ʾ����
        if not found_person:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    # ��ͼ��ת��ΪPIL Image��ʽ����GUI����ʾ
 
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    '''
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)���ⲿ��ʹ�� OpenCV �ĺ�����ͼ�� frame ����ɫ�ռ�� BGR ת��Ϊ RGB
    Image.fromarray(...)����ת����� numpy ���飨��������ɫ�ռ�ת�����ͼ�����ݣ�ת��Ϊ PIL ��Python Imaging Library���� Image ����
    
    ������һϵ��ת��ͨ����Ϊ���� Tkinter ͼ�ν�������ʾ���� OpenCV �����ͼ��
    Tkinter �� Python ��׼�����Դ���һ�����ڴ���ͼ���û����棨GUI���Ŀ�
    '''
    photo = ImageTk.PhotoImage(image)
    '''
    ���д��� photo = ImageTk.PhotoImage(image) ������
    �ǽ�һ�� PIL ��Python Imaging Library���� Image ���� image ת��Ϊ Tkinter �ܹ�ʶ�����ʾ��ͼ���ʽ PhotoImage �������丳ֵ������ photo ��
    '''
 
    # ���±�ǩ����ʾͼ��
    image_label.configure(image=photo)
    image_label.image = photo
 
    # ����GUI�¼��Ա���������
    root.update()
#�ر�����ͷ�����ٴ���
cap.release()
cv2.destroyAllWindows()