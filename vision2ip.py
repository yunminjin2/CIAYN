import json
from ultralytics import YOLO

import os
import numpy as np
import cv2
from collections import defaultdict

import util
import detection
from car import IPCar


class GPS2Pixel:
    def __init__(self, grid_data):
        self.grid_data = grid_data
        self.grid_h, self.grid_w = grid_data['length'][0], grid_data['length'][1]

        self.create_grid_map()
        self.create_gps_map()

    def gps2img(self, gps):
        car_gps = np.float32(gps)
        trans_gps = np.squeeze(self.gps2img_map @ np.expand_dims(np.append(car_gps, 1), axis=1))
        return trans_gps[:2] / trans_gps[-1]

    def create_gps_map(self,):
        gps_data = np.float32(grid_data['GPS'])
        road_square = np.float32(self.grid_data['road_square'])

        self.gps2img_map = cv2.getPerspectiveTransform(gps_data, road_square)
        self.img2gps_map = cv2.getPerspectiveTransform(road_square, gps_data)

    def create_grid_map(self):
        road_square = np.float32(self.grid_data['road_square'])
        trg = np.float32([
            [3/8*self.grid_h, 3/8*self.grid_w],
            [5/8*self.grid_h, 3/8*self.grid_w], 
            [5/8*self.grid_h, 5/8*self.grid_w], 
            [3/8*self.grid_h, 5/8*self.grid_w]]
        )

        self.img2grid_map = cv2.getPerspectiveTransform(road_square, trg)
        self.grid2img_map = cv2.getPerspectiveTransform(trg, road_square)

class Vision2IP(GPS2Pixel):
    def __init__(self, img, grid_data, received_data):
        super().__init__(grid_data)
        self.received_data = received_data
        self.img = img
        self.h, self.w, self.c = img.shape

    def convert_gps2pixel(self):
        trans_car_data = car_data.copy()
        for car_id in car_data.keys():
            pixel_coor = np.array(self.gps2img(car_data[car_id]['gps']), dtype=np.uint32)
            trans_car_data[car_id]['pixel'] = pixel_coor.tolist()

        self.received_data_with_pixel = trans_car_data

    def match_coordinate(self, res):
        def get_bbox(bbox):
            return np.array([int(bbox[0]*self.w), int(bbox[1]*self.h)]), np.array([int(bbox[2]*self.w), int(bbox[3]*self.h)])
        def center_box_coor(p1, p2):
            return (p1 + p2) // 2

        def search_nearest_gps(bbox_center, received_data):
            min_dist = 100000
            car_id = '0'
            for each_car in received_data.keys():
                dist = np.linalg.norm(bbox_center - received_data[each_car]['pixel'])
                if dist < min_dist:
                    min_dist = dist
                    car_id = each_car

            return car_id


        label_dict = res[0].names   
        res_data = dict()
        
        boxes = res[0].boxes.cpu().numpy()

        for idx in range(boxes.shape[0]):
            each_bbox = boxes.xyxyn[idx]
            p1, p2 = get_bbox(each_bbox)
            bbox_center = center_box_coor(p1, p2)

            class_id = int(boxes.cls[idx])
            class_name = label_dict[class_id]

            car_id = search_nearest_gps(bbox_center, self.received_data_with_pixel)
            
            res_data[car_id] = IPCar(
                ip=self.received_data_with_pixel[car_id]['ip'], 
                label_id=class_id, 
                gps=self.received_data_with_pixel[car_id]['gps'], 
                gps_pixel=self.received_data_with_pixel[car_id]['pixel'], 
                bbox=[p1, p2], 
                label=class_name
            )


        return res_data
            
def plot(img, car_dict, bbox=True, labels=True):
    h, w, c = img.shape

    if bbox:
        for car_id in car_dict.keys():
            car = car_dict[car_id]
            p1, p2 = car.bbox[0], car.bbox[1]

            cv2.rectangle(img, p1, p2, detection.palette[car.label_id].tolist(), thickness=1, lineType=cv2.LINE_AA)
            if labels:
                img = detection.draw_labels(img, p1, p2, f'{car.type}:{car.ip}', car.label_id)
            img = cv2.circle(img, car_dict[car_id].gps_pixel, 5, (0, 0, 255), 5)

    return img

if __name__=='__main__':
    road_path = os.path.join('SungBock', 'SB002')
    path_bundle = util.path_load(road_path)
    input_im, car_data, grid_data = util.load_input(path_bundle)

    cls_model = YOLO('yolov8l.pt')

    res = cls_model(input_im)

    mapper = Vision2IP(input_im, grid_data, car_data)
    mapper.convert_gps2pixel()

    car_dict = mapper.match_coordinate(res)
    

    ## plot
    res_img = plot(input_im, car_dict)

    cv2.imshow('img', res_img)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join('ip_img', road_path + '.jpg'), res_img)