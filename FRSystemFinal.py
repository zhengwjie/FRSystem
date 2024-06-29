import os
import sys

import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QFileDialog, QAbstractItemView, QHeaderView
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer
from FRSystem3 import Ui_FRSystem
from cv2 import VideoCapture
from FaceDetection import FaceDetect
from gui_utils import setPixmap_Wrapper
from PersonManagement import PersonManagement
import cv2
from PyQt5.QtGui import QStandardItem,QStandardItemModel
from FaceRecognition import FaceRecognition
class MainWindow(QWidget,Ui_FRSystem):
    def __init__(self):
        super().__init__()

        self.cap=VideoCapture()
        self.read_vedio_timer=QTimer()
        # 创建人脸检测算法
        self.face_detector=FaceDetect()
        self.face_recognizer=FaceRecognition()
        # 每10次存储一次人脸数据
        self.cnt=0
        # 人的身份信息数据库管理对象
        self.identityManagement=PersonManagement()
        self.face_path_to_recognition=None


        self.setupUi(self)
        self.slot_init()


    def slot_init(self):
        # 注册页面切换事件
        self.pushButton_11.clicked.connect(self.stacked_widget_btn_clicked_11)
        self.pushButton_21.clicked.connect(self.stacked_widget_btn_clicked_21)
        self.pushButton_31.clicked.connect(self.stacked_widget_btn_clicked_31)
        self.pushButton_41.clicked.connect(self.stacked_widget_btn_clicked_41)
        self.pushButton_51.clicked.connect(self.stacked_widget_btn_clicked_51)

    #登录和取消按钮
        self.pushButton.clicked.connect(self.login_in)
        self.pushButton_2.clicked.connect(self.close)

    # 信息录入事件
        self.pushButton_3.clicked.connect(self.start_collect_face)
        self.pushButton_4.clicked.connect(self.stop_collect_face)
        self.pushButton_5.clicked.connect(self.save_info)
        self.read_vedio_timer.timeout.connect(self.show_camera_and_detected_face)

    # 信息查看两个按钮事件
        self.pushButton_8.clicked.connect(self.open_human_face)
        self.pushButton_7.clicked.connect(self.query_human_info)
    # 人脸识别demo
    #    1.打开图片文件，并显示到qlabel上
    #    2.开始识别人脸的信息
        self.pushButton_9.clicked.connect(self.open_image_file_and_show)
        self.pushButton_6.clicked.connect(self.start_face_recognition)

    # 修改设置
        self.pushButton_10.clicked.connect(self.change_root_path)
        self.comboBox.currentIndexChanged.connect(self.change_feature_extract_model)
        self.comboBox_2.currentIndexChanged.connect(self.change_human_face_detection_algorithm)
        self.comboBox_3.currentIndexChanged.connect(self.change_human_face_recognition_algorithm)
    def change_root_path(self):
        # 弹出选择文件对话框
        dir=QFileDialog.getExistingDirectory(
            self,
            "选择人脸图像保存路径",
            "./"
        )
        if dir=="":
            print("取消选择")
            QMessageBox.information(self,"提示","你已取消选择",QMessageBox.Ok)
            return
        self.textEdit.setText(dir)
        self.identityManagement.faceManagement.set_root(dir)
    def change_feature_extract_model(self):
        # pca_model
        # facenet_model
        feature_extract_model_name=self.comboBox.currentText()
        self.identityManagement.faceManagement.set_model(feature_extract_model_name)

    def change_human_face_detection_algorithm(self):
        algorithm_name=self.comboBox_2.currentText()
        self.face_detector.set_model(algorithm_name)
        pass
    def change_human_face_recognition_algorithm(self):
        algorithm_name = self.comboBox_3.currentText()
        self.face_recognizer.set_model(algorithm_name)
        pass

    def open_image_file_and_show(self):
        file,fileType=QFileDialog.getOpenFileName(
            self,
            "选取人脸图像",
            "./",
            "All Files (*);;PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        if file=="":
            print("取消选择")
            return
        self.face_path_to_recognition=file
        pixmap=QPixmap(file)
        self.label_12.setPixmap(pixmap)
        self.label_12.setScaledContents(True)
    # 开始进行人脸识别
    def start_face_recognition(self):
        # 使用opencv进行人脸识别
        model=self.face_recognizer.get_model()
        imgs,labels=self.face_recognizer.get_dataset(self.identityManagement.faceManagement.root)
        self.face_recognizer.train(model,imgs,labels)
        label, confidence,stranger=self.face_recognizer.predict(model,self.face_path_to_recognition)
        print(label,confidence)
        print(type(labels))
        print(type(confidence))
    #     查询数据库，找到人的信息
        if stranger:
            self.textBrowser.setText("系统识别到这是一个陌生人脸图像！！！")
            pass
        else:
            # 查询数据库
            id,name,age=self.identityManagement.query_person(label)
            print(id,name,age)
            self.textBrowser.setText("系统识别到这是一张身份id为 {} ,姓名为 {} ,年龄为 {} 的人脸图像！".format(id,name,age))
            pass


    def stacked_widget_btn_clicked_11(self):
        self.stackedWidget.setCurrentIndex(0)
    def stacked_widget_btn_clicked_21(self):
        self.stackedWidget.setCurrentIndex(1)

    def stacked_widget_btn_clicked_31(self):
        self.stackedWidget.setCurrentIndex(2)

    def stacked_widget_btn_clicked_41(self):
        self.stackedWidget.setCurrentIndex(3)

    def stacked_widget_btn_clicked_51(self):
        self.stackedWidget.setCurrentIndex(4)

    def login_in(self):
    #    获取文本并且检查是否正确
        user_name=self.lineEdit.text()
        passwd=self.lineEdit_2.text()
        print(user_name)
        print(passwd)
        if user_name=='root' and passwd=='root':
            QMessageBox.information(self,"提示","登录成功",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
            self.stackedWidget.setCurrentIndex(1)
        else:
            QMessageBox.information(self, "错误", "登录失败", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            self.lineEdit.clear()
            self.lineEdit_2.clear()
    def login_quit(self):
        # self.stackedWidget.setCurrentIndex(1)
        self.close()
        pass

#     三个按钮事件
    def start_collect_face(self):
        # 打开摄像头
        # 进行人脸检测
        # 显示到界面上
        # 根据用户的id
        # 每10帧保存一次
        # 把检测到的人脸保存到文件夹中
        # 正在采集图像，无需打开摄像头
        if self.read_vedio_timer.isActive():
            return
        # 打开摄像头
        flag = self.cap.open(0)
        if flag == False:
            # 提示错误信息
            msg = QMessageBox.warning(self, 'warning', "打开失败",
                                                buttons=QMessageBox.Ok)
        else:
            # 每隔30ms 我们读取摄像头中的数据一次
            self.read_vedio_timer.start(50)

    def show_camera_and_detected_face(self):
        # 超时之后
        # 进行人脸检测
        faces,img=self.face_detector.detect_face_by_video_cap(self.cap)

        # 把检测结果展示到label上
        # 创建QImage对象
        # 创建QPixmap对象
        #
        # 展示摄像头数据
        # self.label_9.setPixmap()
        # 展示检测到的人脸照片
        # self.label_10.setPixmap()
        setPixmap_Wrapper(self.label_9, img)
        if len(faces)==0:
            return
        face = faces[0].copy()
        cv2.imwrite("face.png",cv2.cvtColor(face,cv2.COLOR_RGB2BGR))

        setPixmap_Wrapper(self.label_10, face)
        try:
            user_id=int(self.lineEdit_3.text())
        except:
            return
        self.cnt += 1
        if self.cnt % 10 == 0:
            self.identityManagement.add_face_by_array(user_id,face)
            self.cnt=0

      #     保存图像到数据库中
    def stop_collect_face(self):
        # 关闭摄像头
        # 清除
        self.read_vedio_timer.stop()
        self.cap.release()
        self.label_9.clear()
        self.label_10.clear()
        self.identityManagement.update_feature(int(self.lineEdit_3.text()))

    def save_info(self):
        # 把信息存储在数据库中
        try:
            id=int(self.lineEdit_3.text())
            name=self.lineEdit_4.text()
            age=int(self.lineEdit_5.text())
        except:
            return
        self.identityManagement.add_person(id,name,age)
    # 打开文件夹并且显示在label上
    def open_human_face(self):
        self.stackedWidget_2.setCurrentIndex(0)
        # 弹出一个框，让用户选择文件夹

        # 获取文件路径
        # 把文件展示到QLabel上
        file,fileType=QFileDialog.getOpenFileName(
            self,
            "选取人脸图像",
            self.identityManagement.faceManagement.root,
            "All Files (*);;PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        if file=="":
            print("取消选择")
            return
        # print(file,fileType)
        pixmap=QPixmap(file)
        # pixmap = pixmap.scaled(200, 200, PyQt6.KeepAspectRatio)  # 缩放图像至200x200像素
        self.label_11.setPixmap(pixmap)
        # label.setPixmap(pixmap)
        self.label_11.setScaledContents(True)

    # 显示所有的人的身份信息
    def query_human_info(self):
        self.stackedWidget_2.setCurrentIndex(1)
        # 查询所有人的信息
        # 使用
        # self.tableWidget
        # 展示所有信息
        # 获取行数
        ids=os.listdir(self.identityManagement.faceManagement.root)
        # self.identityManagement.query_person(row_cnt,3)
        persons=[]
        for id in ids:
            if id is None:
                continue
            person=self.identityManagement.query_person(int(id))
            persons.append(person)

        row_cnt=len(persons)
        model=QStandardItemModel(row_cnt,3)
        model.setHorizontalHeaderLabels(["身份id","姓名","年龄"])
        for row in range(row_cnt):
            for col in range(3):
                item=QStandardItem(str(persons[row][col]))
                model.setItem(row,col,item)
        self.tableView.setModel(model)
        # self.tableView.resizeRowToContents(row_cnt)
        self.tableView.resizeColumnsToContents()
        self.tableView.resizeRowsToContents()
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 设置行列填满窗口
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 使表宽度自适应
        # self.tableView.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 使表高度自适应



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    sys.exit(app.exec_())

