#! /usr/bin/env python
# coding=utf-8

import numpy as np
import core.utils as utils
from core.config import cfg

# NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
# STRIDES         = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
# XYSCALE = cfg.YOLO.XYSCALE
# ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0 
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

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