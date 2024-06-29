from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel

import numpy as np

# 自适应展示图片
def setPixmap_Wrapper(label_obj:QLabel,img_obj:np.ndarray):
    label_width=label_obj.geometry().width()
    label_height=label_obj.geometry().height()
    img_height=img_obj.shape[0]
    img_width=img_obj.shape[1]

    # print(label_width,label_height,img_height,img_width)

    ratio1,ratio2=img_width/label_width,img_height/label_height
    ratio=max(ratio1,ratio2)
    # im_np = np.transpose(img_obj, (1, 0, 2))

    picture = QtGui.QImage(img_obj.data, img_width, img_height,3*img_width, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(picture)
    # 按照缩放比例自适应 label 显示
    pixmap.setDevicePixelRatio(ratio)
    label_obj.setPixmap(pixmap)
    return
