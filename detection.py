from ultralytics import YOLO
import cv2
import torch
import os
import numpy as np

import random

cls_model = YOLO('yolov8l.pt')

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                    dtype=np.uint8)


def load_input(path):
    im = cv2.imread(path)  # OpenCV
    
    # t = cv2.resize(im, (640, 640))
    print('img loaded')

    return im



def draw_labels(img, p1, p2, label, int_label):
    w, h = cv2.getTextSize(label, 0, fontScale=3 / 3, thickness=1)[0]  # text width, height
    outside = p1[1] - h >= 3
    cv2.putText(img,
        label, 
        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        0,
        1,
        palette[int_label].tolist(),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    return img


def plot(img, res, bbox=True, labels=True):
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
    path = os.path.join('img', 'SJ.jpg')

    im_input = load_input(path)
    h, w, c = im_input.shape
    res = cls_model(im_input)

    res_img = plot(im_input, res)
    
    cv2.imshow('res', res_img)
    cv2.waitKey(0)