#================================================================
#
#   File name   : yolov4.py
#   Author      : PyLessons
#   Created date: 2020-09-31
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : main yolov3 & yolov4 functions
#
#================================================================
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
# from tensorflow.keras.regularizers import l2
# from app.visao.yolov3.configs import *

# STRIDES         = np.array(YOLO_STRIDES)
# ANCHORS         = (np.array(YOLO_ANCHORS).T/STRIDES).T

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape=(416, 416), image_shape=(416, 416)):
    # Deixar só a classe com o maior score
    scores_max = np.amax(scores, axis=1)
    # print(scores_max.shape)
    image_h = input_shape[0]
    image_w = input_shape[1]

    # Indices das detecções válidas
    indices = np.argwhere(scores_max >= score_threshold)
    # print(indices)
    # print(box_xywh.shape)

    # (x1, y1), (x2, y2), score, class
    bboxes = []
    for item in indices:
        box = []
        # Coordenadas
        x = box_xywh[item][0][0] 
        y = box_xywh[item][0][1]
        w = box_xywh[item][0][2] 
        h = box_xywh[item][0][3]

        # print(x)
        # print(y)
        # print(w)
        # print(h)

        xmin = (x-int(w/2))
        ymin = (y-int(h/2))
        xmax = (x+int(w/2))
        ymax = (y+int(h/2))

        box.append(xmin)
        box.append(ymin)
        box.append(xmax)
        box.append(ymax)
        # Score
        box.append(scores_max[item][0])
        # Classe
        box.append(np.argmax(scores[item]))
        bboxes.append(box)
    bboxes = np.asarray(bboxes)
    # print(bboxes)
    # print(bboxes.shape)

    return bboxes
