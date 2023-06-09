from ultralytics import YOLO
import cv2
import os
import numpy as np

import random

import util
from util import palette


def draw_labels(img, p1, p2, label, int_label):
    w, h = cv2.getTextSize(label, 0, fontScale=3 / 3, thickness=1)[0]  # text width, height
    outside = p1[1] - h >= 3
    cv2.putText(img,
        label, 
        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        0,
        1,
        palette[int_label].tolist(),
        thickness=2,
        lineType=cv2.LINE_AA
    )
    return img


def plot(img, res, bbox=True, labels=True):
    h, w, c = img.shape
    label_dict = res[0].names
    if bbox:
        boxes = res[0].boxes.cpu().numpy()
        for idx in range(boxes.shape[0]):
            each_bbox = boxes.xyxyn[idx]
            class_id = int(boxes.cls[idx])
            class_name = label_dict[class_id]

            p1, p2 = (int(each_bbox[0]*w), int(each_bbox[1]*h)), (int(each_bbox[2]*w), int(each_bbox[3]*h))
            cv2.rectangle(img, p1, p2, palette[class_id].tolist(), thickness=1, lineType=cv2.LINE_AA)
            if labels:
                # import pdb; pdb.set_trace()
                img = draw_labels(img, p1, p2, class_name, class_id)

    return img


if __name__ == '__main__':
    cls_model = YOLO('yolov8l.pt')
    path = os.path.join('img', 'GwangGuo', 'GW001.jpg')

    im_input, _, _ = util.load_input(path)
    res = cls_model(im_input)

    res_img = plot(im_input, res)
    
    cv2.imshow('res', res_img)
    cv2.waitKey(0)