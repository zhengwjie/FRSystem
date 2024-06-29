# 使用SQLite实现一个人脸管理系统，支持增删改查，
# 人脸图像保存在文件夹中，每个人创建一个文件夹，

# 通过pca模型抽取用户的人脸特征（如果一个人有多个照片，使用特征的均值表示该人脸）

# 每个人有一个id，name，age等属性，同时，

# data/{id}/1.png 2.png   => np.ndarray()  joblib.load
#          id.pkz
# data/2/a.png b.png

# id name age

# 增加  传入参数 (id  img)-> 写入系统   =>更新特征
# 删除  (id)
# 删除  (ID  图片)  1.文件夹删除图片  2.更新特征
# 修改  (id  A=>B)    1.文件夹修改图片  2.更新特征
# 查询  (id)     name age 特征向量  所有人脸图像

# pca模型加载与保存


import joblib
import os
import shutil

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import cv2
import sqlite3
# 定义人
from facenet_pytorch import InceptionResnetV1

class Person:
    def __int__(self,id,name,age,img_paths):
        # 读取数据
        pass
#     根据id获取信息


class FaceManagement:
    # 每个人的人脸照片都保存在./data/{id}下
    # 特征命名为./data/{id}/id.pkl
    # 抽特征的模型使用pca模型
    model_lib={
        "facenet_model":InceptionResnetV1(pretrained='vggface2').eval()
    }
    def __init__(self,root,pca_model_path):
        self.root=root
        self.pca_model_path=pca_model_path
        self.pca_model:PCA=joblib.load(pca_model_path)
        FaceManagement.model_lib['pca_model']=self.pca_model
        self.model=self.pca_model
        self.model_name='pca_model'

    def set_root(self,root_dir):
        self.root=root_dir

    def set_model(self,model_name):
        self.model_name=model_name
        self.model=FaceManagement.model_lib[model_name]
        # 重写
        pass


    def get_face_num(self,id):
        face_dir = os.path.join(self.root, str(id))
        if not os.path.exists(face_dir):
            return 0
        files=os.listdir(face_dir)
        n=len(files)
        return n-1
    def get_collected_face_num(self,id):
        face_dir = os.path.join(self.root, str(id))
        if not os.path.exists(face_dir):
            return 0
        files=os.listdir(face_dir)
        png_files=[item for item in files if item.endswith(".png")]
        return len(png_files)

    def feature_extract(self,image_path):
        if self.model_name=='pca_model':
            return self.feature_extract_pca(image_path)
        else:
            return self.feature_extract_facenet(image_path)

    def feature_extract_pca(self,image_path):
        img=cv2.imread(image_path)
        img.resize((64,64))
        img=img.reshape((1,-1))
        img_feature=self.pca_model.transform(img)
        return img_feature
    def feature_extract_facenet(self,image_path):
        # to do

        img=cv2.imread(image_path)

        img.resize((64,64))
        img=img.reshape((1,-1))
        img_feature=self.pca_model.transform(img)
        return img_feature


    def get_cur_feature(self,id):
        face_dir = os.path.join(self.root, str(id))
        cur_feature_path=os.path.join(face_dir,str(id)+'.pkl')
        cur_feature=joblib.load(cur_feature_path)
        return cur_feature

    # 增加人脸
    def add_face(self,id,image_path):
        if not os.path.exists(image_path):
            return -1
        # add_face
        # 获取到人脸的目录
        face_dir=os.path.join(self.root,str(id))
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
        # root/{id}/1.png
        # a/b/1.png
        (path, filename) = os.path.split(image_path)
        shutil.copyfile(image_path, os.path.join(face_dir,filename))

        # 提取特征，获取特征向量
        img_feature=self.feature_extract(image_path)
        # 更新特征向量
        m = self.get_face_num(id)

        old_feature_path=os.path.join(face_dir,str(id)+'.pkl')
        if m==0:
            old_feature=np.zeros(img_feature.shape)
        else:
            old_feature=self.get_cur_feature(id)
        # m   (m*old_feature+img_feature)/(m+1)

        new_feature=(old_feature*m+img_feature)/(m+1)
        joblib.dump(new_feature,old_feature_path)
        return 0
        # (x*old_featurn+new_feature)/(x+1)

    def add_face_by_array(self,id,img_array):
        img_bgr=cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # 需要用户的id
        face_dir=os.path.join(self.root,str(id))
        face_file_name=str(self.get_collected_face_num(id))+".png"
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
        cv2.imwrite(os.path.join(face_dir,face_file_name), img_bgr)
    def update_feature(self,id):
        face_dir=os.path.join(self.root,str(id))
        # 路径不存在，直接返回
        if not os.path.exists(face_dir):
            return
        feature_list=[]
        for file in os.listdir(face_dir):
            # print(file)
            if file.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                feature_list.append(self.feature_extract(os.path.join(face_dir,file)))
        concated_feature=np.concatenate(feature_list,axis=0)
        mean_feature=np.mean(concated_feature, axis=0)
        feature_path = os.path.join(face_dir, str(id) + '.pkl')
        joblib.dump(mean_feature,feature_path)

    # 删除
    def delete_faces(self, id):
        # 删除
        face_dir = os.path.join(self.root, str(id))
        shutil.rmtree(face_dir)

        # os.removedirs(face_dir)
        return 0

    def delete_face(self,id,image_name):
        face_dir = os.path.join(self.root, str(id))
        img_path = os.path.join(face_dir, image_name)
        if not os.path.exists(img_path):
            return 1

        img_feature=self.feature_extract(img_path)
        m=self.get_face_num(id)
        mean_feature=self.get_cur_feature(id)
        if m==1:
            new_feature=img_feature
        else:
           new_feature=(m*mean_feature-img_feature)/(m-1)
    #     写到文件中
        joblib.dump(new_feature,os.path.join(face_dir,str(id)+'.pkl'))
    #  m   m  (m*mean_feature-feature)/(m-1)
    #     先更新特征
    #     删除照片
        img_path=os.path.join(face_dir,image_name)
        os.remove(img_path)
        return 0

    # 修改： 1. 删除   2.增加
    def modify_face(self, src_img_name,dst_img_path,id):
        self.delete_face(id,src_img_name)
        self.add_face(id,dst_img_path)
        return 0

    # 查询
    # 图片的路径以及特征向量
    def query_face(self, id):
        face_dir = os.path.join(self.root, str(id))
        if not os.path.exists(face_dir):
            return None,None
        files = os.listdir(face_dir)
        # self.root/{id}/1.png
        # 1.png  {id}.pkl
        file_list=[]
        for file in files:
            if file==(str(id)+".pkl"):
                continue
            f=os.path.join(face_dir,file)
            file_list.append(f)
        feature=self.get_cur_feature(id)
        return file_list,feature

    def get_all_features(self):
        all_features=[]
        files=os.listdir(self.root)
        for id in files:
        #     root/{id}/{id}.pkl
            feature_path=os.path.join(self.root,id,id+".pkl")
            feature=joblib.load(feature_path)
            all_features.append(feature)
        return all_features,files


class IdentityManagement:
    # sqlite数据库管理数据
    def __init__(self,database_path):
        self.conn = sqlite3.connect(database_path)
        self.c = self.conn.cursor()
        # 创建一个表
        self.c.execute('''
        create table if not exists person(
        id int primary key not null,
        name text not null,
        age int not null
        )
        ''')
        self.conn.commit()
        # cursor.execute('''
        # CREATE TABLE company(
        # id int primary key not null,
        # name text not null,
        # age int not null,
        # address char(50),
        # salary real);
        # ''');
        pass
    def add_person(self, id, name,age):
        try:
            self.c.execute('''insert into person values({},'{}',{})'''.format(id, name, age)
            )
        except:
            return -1
        self.conn.commit()
        return 0
    def query_person(self, id):
        res=self.c.execute('''
        select id,name,age from person where id={}'''.format(id))
        res_list=[]
        # print(res)
        for row in res:
            res_list.append((row[0],row[1],row[2]))
        self.conn.commit()
        if len(res_list)>0:
            return res_list[0]
        return None

    # UPDATE table_name
    # SET column1 = value1, column2 = value2...., columnN = valueN
    # WHERE [condition];
    def modify_person(self,id,name,age):
        try:
            self.c.execute('''update person set name='{}',age={} where id={}'''.format(name, age,id))
            self.conn.commit()
        except:
            return -1
        return 0



    # DELETE
    # FROM
    # table_name
    # WHERE[condition];
    def delete_person(self, id):
        try:
            self.c.execute('''delete from person where id={}'''.format(id))
            self.conn.commit()
        except:
            return -1
        return 0

class PersonManagement:
    def __init__(self,root="./data",pca_model_path='./pca.model',database_path="./test.db"):
        self.faceManagement=FaceManagement(root,pca_model_path)
        self.identityManagement=IdentityManagement(database_path)
    def add_face(self,id,image_path):
        return self.faceManagement.add_face(id,image_path)

    def add_face_by_array(self, id, img_array):
        return self.faceManagement.add_face_by_array(id,img_array)

    def update_feature(self, id):
        return self.faceManagement.update_feature(id)

    def delete_faces(self, id):
        return self.faceManagement.delete_faces(id)

    def delete_face(self, id, image_name):
        return self.faceManagement.delete_face(id,image_name)

    def modify_face(self, src_img_name, dst_img_path, id):
        return self.faceManagement.modify_face(src_img_name,dst_img_path,id)
    def query_face(self, id):
        return self.faceManagement.query_face(id)
    def add_person(self, id, name,age):
        return self.identityManagement.add_person(id,name,age)

    def query_person(self, id):
        return self.identityManagement.query_person(id)
    def modify_person(self, id, name, age):
        return self.identityManagement.modify_person(id,name,age)

    def delete_person(self, id):
        return self.identityManagement.delete_person(id)

    def get_identity(self,img_path):
        # 1. 读取图片
        # 2. 抽取特征
        # 3. 和现有的特征进行比对
        # matrix=cv2.imread(img_path)
        #
        y=self.faceManagement.feature_extract(img_path)
        # 把现有的特征取出来
        features,file_list=self.faceManagement.get_all_features()
        features=np.vstack(features)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(features)

        distances, indices = nbrs.kneighbors(y)
        # 比对特征
        id=int(file_list[indices[0][0]])
        # 获取特征的id
        person_info=self.identityManagement.query_person(id)
        return person_info

if __name__=='__main__':
    # root,pca_model_path,database_path
    root_dir="./data"
    pca_model_path='./pca.model'
    database_path="./test.db"
    personSystem=PersonManagement(root_dir,pca_model_path,database_path)

    # personSystem.add_face(1,"img.png")
    # personSystem.add_face(2,'img.png')
    # personSystem.add_face(2, 'face.PNG')

    # personSystem.delete_faces(1)

    # personSystem.delete_face(2,'img.png')

    # personSystem.modify_face("img.png","face.PNG",2)

    # personSystem.add_face(2,"1.png")
    # personSystem.add_face(2, "img.png")
    #
    # file_list,feature=personSystem.query_face(1)
    # print(file_list)
    # print(feature)
    # print(file_list)
    # print(feature.shape)
    # print(res)
    id=2
    name="Danny"
    age=20
    # '''insert into person values({},{},{})'''.format(id,name,age)
    # personSystem.add_face(1,"1.png")
    personSystem.add_person(2,'Tony',23)

    #
    #
    # # person=personSystem.query_person(2)
    # # id,name,age=person
    # # print(id)
    # # print(name)
    # # print(age)
    #
    # person = personSystem.query_person(3)
    # print(person)
    #
    # personSystem.modify_person(2,"Tony",23)
    #
    # person=personSystem.query_person(2)
    # id,name,age=person
    # print(id)
    # print(name)
    # print(age)
    # personSystem.delete_person(2)

    # img   ----->     id,name,age

    person_info=personSystem.get_identity("1.png")
    print(person_info)
    # # 读取用户信息
    # faceSystem=FaceManagement('./a','./model.name')
    #
    # faceSystem.add_face()
    #
    # personSystem=IdentityManagement("./a")
    #
    # personSystem.add_person()
    #
    #
    #
    # a=System()
    # a.add_face()
    # a.add_person()


    pass