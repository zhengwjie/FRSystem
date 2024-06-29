


# 实现一些人脸检测的函数
import cv2
import dlib
class FaceDetect:
    model_lib={
        "haar":"haarcascade_frontalface_default.xml",
    }
    # 类的创建
    def __init__(self,model_name='haar'):
        self.model_name=model_name
        # 创建分类器
        self.classifier=cv2.CascadeClassifier(cv2.data.haarcascades+FaceDetect.model_lib[model_name])
        if model_name=='dlib':
            self.classifier= dlib.get_frontal_face_detector()
    def set_model(self,model_name):
        self.model_name=model_name
        if model_name=='haar':
            self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + FaceDetect.model_lib[model_name])
        else:
            self.classifier = dlib.get_frontal_face_detector()

    def detect_face_by_image_path(self,img_path):
        img = cv2.imread(img_path)
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.detect_face_by_img(show)

    def detect_face_by_video_cap(self,cap):
        # 传入一个cap对象
        flag, img = cap.read()
        # 对图像进行处理
        # BGR转为RGB
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.detect_face_by_img(show)

    # 传入一个多维矩阵
    def detect_face_by_img(self,img):
        if self.model_name=='haar':
            return self.detect_face_by_img_haar(img)
        else:
            return self.detect_face_by_img_dlib(img)

    def detect_face_by_img_haar(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #     # 检测人脸
        faces = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        cliped_faces = []
        # 绘制检测到的人脸
        for (x, y, w, h) in faces:

            new_faces = img[y :y + h , x:x + w].copy()
            #
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cliped_faces.append(new_faces)

        return cliped_faces,img

    def detect_face_by_img_dlib(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.classifier(gray)

        cliped_faces = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            new_faces = img[y:y + h, x:x + w].copy()
            #
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cliped_faces.append(new_faces)

        return cliped_faces,img




# def face_detect(img_path):
#     # 加载预训练的人脸检测分类器
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#
#     # 读取图像
#     img = cv2.imread('/Users/zwj/PycharmProjects/LeNet/course_examples/day5/face.PNG')
#     # print(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # 检测人脸
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     cliped_faces=[]
#     # 绘制检测到的人脸
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         print(x, y, w, h)
#         new_faces = img[y - 2:y + h + 2, x - 2:x + w + 2]
#         # cv2.imwrite("res.png", new_faces)
#         cliped_faces.append(new_faces)
#
#     return cliped_faces
#