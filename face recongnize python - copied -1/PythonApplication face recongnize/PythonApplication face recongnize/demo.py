#-*-coding:GBK -*-
import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk,ImageDraw
import numpy as np
 
from PIL import ImageFont
 
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
 
##加载自定义字体
font = ImageFont.truetype(r"C:\Users\agaci\Desktop\YaHei.Consolas.1.11b.ttf", size=30)
 
# 加载Haar Cascade分类器
face_cascade = cv2.CascadeClassifier(r"C:\Users\agaci\Desktop\face recongnize python\haarcascade_frontalface_default.xml")
 
# 创建GUI窗口
root = tk.Tk()#Tk 类代表了应用程序的主窗口。通过创建 Tk 类的实例并将其赋值给 root 变量，我们就建立了应用程序的主界面框架。
root.geometry('640x480')#设置窗口大小
root.title('face recongnize')#设置窗口标题
'''
        root = tk.Tk() 这行代码创建了一个顶级窗口对象 root 。
        root.geometry('640x480') 用于设置窗口的初始大小为 640 像素宽和 480 像素高。
        root.title('人脸识别') 则为这个窗口设置了标题为“人脸识别”
'''
# 创建标签用于显示图像
image_label = tk.Label(root)
image_label.pack()
'''
tk.Label(root)：Label 是 tkinter 库中的一个控件类，用于创建标签。
root 作为参数传递给 Label 的构造函数，表示这个标签将放置在 root 这个主窗口中。

pack() 是一种几何管理器方法。
它的作用是自动将控件放置在窗口中，并根据窗口的可用空间和其他已放置的控件来确定控件的位置和大小。

'''
# 打开摄像头并捕获实时图像
cap = cv2.VideoCapture(0)
'''
VideoCapture 类用于从摄像头、视频文件或其他视频源中获取视频帧。
在这个例子中，参数 0 表示使用默认的摄像头（通常是计算机内置的摄像头）

如果要从视频文件中读取视频，
可以将文件路径作为参数传递，例如 cap = cv2.VideoCapture('video.mp4')
'''
# 创建 PhotoImage 对象
photo = None
 
# 读取person文件夹中的图像和姓名
person_images = []
person_names = []
for filename in os.listdir(r"C:\Users\agaci\Desktop\face recongnize python\person"):
    if filename.endswith('.jpg'):
        # 使用utf-8编码打开文件
        with open(os.path.join(r"C:\Users\agaci\Desktop\face recongnize python\person", filename), 'rb') as f:
            person_images.append(cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR))
        person_names.append(os.path.splitext(filename)[0])
 
 
# 循环处理摄像头捕获的图像
while True:
    ret, frame = cap.read()
    if not ret:# ret 通常是一个布尔值（True 或 False）
        break
 
    # 转换图像格式以进行人脸检测
    # 使用 OpenCV 库中的 cv2.cvtColor 函数将输入的图像 frame 从 BGR 颜色空间转换为灰度（GRAY）颜色空间，并将结果存储在变量 gray 中
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # 使用Haar Cascade分类器检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    '''
    使用 OpenCV 中的 face_cascade 对象的 detectMultiScale 方法在灰度图像 gray 中检测人脸
    
    detectMultiScale 方法会返回一个包含检测到的人脸区域的矩形坐标的列表。
    例如，如果在图像中成功检测到了两个人脸，faces 可能会是类似 [(x1, y1, w1, h1), (x2, y2, w2, h2)] 的形式，其中 (x, y) 是矩形左上角的坐标，(w, h) 分别是矩形的宽度和高度。
    
    scaleFactor=1.1 表示在图像金字塔中每次缩放图像的比例。它控制了检测窗口在不同尺度上的缩放程度，值越大，检测速度越快，但可能会漏掉一些较小的人脸；值越小，检测更精细，但计算量会增加。
    minNeighbors=5 表示每个候选矩形区域应该保留的相邻矩形的数量。这个参数用于控制检测到的人脸的准确性。值越大，误检率越低，但可能会漏检一些较难检测的人脸；值越小，可能会检测到更多的误检人脸。
    
    '''
    # 在图像中框出检测到的人脸
    for (x, y, w, h) in faces:
        # 检查人脸是否属于person文件夹中的某个人
        found_person = False
        for i in range(len(person_images)):
            person_image = person_images[i]
            person_name = person_names[i]
            # 将person图像转换为灰度图像以进行比较
            person_gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
            # 检查是否存在与person图像相匹配的人脸
            match = cv2.matchTemplate(gray[y:y + h, x:x + w], person_gray, cv2.TM_CCOEFF_NORMED)
            '''
                gray[y:y + h, x:x + w]：这是从灰度图像 gray 中提取的当前检测到的人脸区域
                person_gray：这是从 person 文件夹中获取的某个人的灰度图像，用于与当前检测到的人脸区域进行匹配比较
                cv2.TM_CCOEFF_NORMED：这是指定的匹配方法。在这种情况下，使用的是“归一化的相关系数”方法.这种方法会计算两个图像区域之间的相关性，并将结果归一化到 [-1, 1] 的范围内。值越接近 1 表示匹配度越高，值越接近 -1 表示匹配度越低
            '''   
            if match.max() > 0.8:
                print(person_name)
                found_person = True
                # 在图像中框出人脸并显示姓名
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                '''
                frame：要绘制矩形的图像。
                (x, y)：矩形的左上角顶点坐标。
                (x + w, y + h)：矩形的右下角顶点坐标。通过左上角坐标 (x, y) 以及给定的宽度 w 和高度 h 来确定右下角的位置。
                (0, 255, 255)：矩形边框的颜色。在 OpenCV 中，颜色通常以 BGR 顺序表示，这里的 (0, 255, 255) 表示黄色（蓝色通道值为 0 ，绿色通道值为 255 ，红色通道值为 255 ）。
                '''
                # 在图像中框出人脸并显示姓名
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                frame = cv2AddChineseText(frame, person_name, (x + (w/2)-10, y - 30), (0, 255, 255), 30)
                break
 
        # 如果没有找到匹配的人脸，则在图像中框出人脸但不显示姓名
        if not found_person:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    # 将图像转换为PIL Image格式以在GUI中显示
 
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    '''
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)：这部分使用 OpenCV 的函数将图像 frame 的颜色空间从 BGR 转换为 RGB
    Image.fromarray(...)：将转换后的 numpy 数组（即经过颜色空间转换后的图像数据）转换为 PIL （Python Imaging Library）的 Image 对象。
    
    这样的一系列转换通常是为了在 Tkinter 图形界面中显示经过 OpenCV 处理的图像。
    Tkinter 是 Python 标准库中自带的一个用于创建图形用户界面（GUI）的库
    '''
    photo = ImageTk.PhotoImage(image)
    '''
    这行代码 photo = ImageTk.PhotoImage(image) 的作用
    是将一个 PIL （Python Imaging Library）的 Image 对象 image 转换为 Tkinter 能够识别和显示的图像格式 PhotoImage ，并将其赋值给变量 photo 。
    '''
 
    # 更新标签以显示图像
    image_label.configure(image=photo)
    image_label.image = photo
 
    # 处理GUI事件以避免程序挂起
    root.update()
#关闭摄像头并销毁窗口
cap.release()
cv2.destroyAllWindows()