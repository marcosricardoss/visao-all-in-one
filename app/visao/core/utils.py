import numpy as np
from core.config import cfg
import colorsys
import random
import cv2

def load_config(FLAGS):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if FLAGS.model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        elif FLAGS.model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE if FLAGS.model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)

def read_class_names(class_file_name):
    print(class_file_name)
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    print(names)
    return names

def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        # coor[0] = int(coor[0])
        # coor[2] = int(coor[2])
        # coor[1] = int(coor[1])
        # coor[3] = int(coor[3])

        print(coor)
        print(image_h)

        x1 = int(coor[0] * image_h)
        x2 = int(coor[2] * image_h)
        y1 = int(coor[1] * image_w)
        y2 = int(coor[3] * image_w)

        print(x1)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        # bbox_thick = int(0.6 * (image_h + image_w) / 600)
        bbox_thick = 2
        # c1, c2 = (coor[0], coor[2]), (coor[1], coor[3])
        # print((c1, c2))
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick)

        # if show_label:
        #     bbox_mess = '%s: %.2f' % (classes[class_ind], score)
        #     t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        #     c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        #     cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

        #     cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image