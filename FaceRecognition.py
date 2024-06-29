import os

import cv2
import numpy as np


# # EigenFace(PCA，5000以下判断可靠）
# recognizer = cv2.face.EigenFaceRecognizer_create()
# showConfidence(imgPath,recognizer,images,labels)
# # LBPH（局部二值模式直方图，0完全匹配，50以下可接受，80不可靠）
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# showConfidence(imgPath,recognizer,images,labels)
# # Fisher(线判别分析 ， 5000以下判断为可靠）
# recognizer = cv2.face.FisherFaceRecognizer_create()
# showConfidence(imgPath,recognizer,images,labels)

# 人脸识别方法的封装
class FaceRecognition:
    model_lib={
        "eigenface":cv2.face.EigenFaceRecognizer_create(),
        "lbphface":cv2.face.LBPHFaceRecognizer_create(),
        "fisherface":cv2.face.FisherFaceRecognizer_create()
    }
    model_threshhold={
        "eigenface": 5000,
        "lbphface": 50,
        "fisherface": 5000
    }
    def __init__(self,model_name="eigenface"):
        self.model_name=model_name
        try:
            self.model=FaceRecognition.model_lib[model_name]
        except:
            self.model = FaceRecognition.model_lib['eigenface']

    def set_model(self,model_name):
        self.model_name=model_name
        try:
            self.model=FaceRecognition.model_lib[model_name]
        except:
            self.model = FaceRecognition.model_lib['eigenface']

    # 获取数据集
    def get_dataset(self,path):
        print(path)
        ids=os.listdir(path)
        imgs=[]
        labels=[]
        for id in ids:
            int_id=int(id)
            face_dir=os.path.join(path,id)
            id_imgs=os.listdir(face_dir)
            for id_img in id_imgs:
                try:
                    img_arr=cv2.imread(os.path.join(face_dir,id_img),cv2.IMREAD_GRAYSCALE)
                    img_arr=cv2.resize(img_arr,(594,594))
                    imgs.append(img_arr)
                    labels.append(int_id)
                except:
                    pass
        # print(imgs)
        # print(labels)
        return imgs,labels

    def get_model(self):
        return self.model

    def train(self,model,imgs,labels):
        model.train(imgs,np.array(labels))

    def predict(self,model,imgPath):
        predict_image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        predict_image=cv2.resize(predict_image,(594,594))
        # 预测并打印结果
        labels, confidence = model.predict(predict_image)
        stranger=False
        if confidence>FaceRecognition.model_threshhold[self.model_name]:
            stranger=True

        return labels,confidence,stranger


