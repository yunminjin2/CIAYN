import json
from ultralytics import YOLO
import os
import argparse
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
        self.gps_data = np.float32(grid_data['GPS'])
        self.road_square = np.float32(grid_data['road_square'])

        self.create_grid_map()
        self.create_gps_map()

    def gps2img(self, gps):
        car_gps = np.float32(gps)
        trans_gps = np.squeeze(self.gps2img_map @ np.expand_dims(np.append(car_gps, 1), axis=1))
        return trans_gps[:2] / trans_gps[-1]

    def create_gps_map(self,):
        self.gps2img_map = cv2.getPerspectiveTransform(self.gps_data, self.road_square)
        self.img2gps_map = cv2.getPerspectiveTransform(self.road_square, self.gps_data)

    def create_grid_map(self):
        trg = np.float32([
            [3/8*self.grid_h, 3/8*self.grid_w],
            [5/8*self.grid_h, 3/8*self.grid_w], 
            [5/8*self.grid_h, 5/8*self.grid_w], 
            [3/8*self.grid_h, 5/8*self.grid_w]]
        )

        self.img2grid_map = cv2.getPerspectiveTransform(self.road_square, trg)
        self.grid2img_map = cv2.getPerspectiveTransform(trg, self.road_square)



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
            


def clip_point(point, h, w):
    point[0] = np.clip(point[0], 0, h)
    point[1] = np.clip(point[1], 0, w)
    return np.array(point, dtype=np.uint32)



def plot(img, car_dict, threshold, mask, thresholding=False, masking=False, bbox=True, labels=True):
    h, w, c = img.shape
    displayed = False

    num_pixels = 0
    num_cars = 0

    if bbox:
        for car_id in car_dict.keys():
            car = car_dict[car_id]
            p1, p2 = car.bbox[0], car.bbox[1]

            # cv2.rectangle(img, p1, p2, detection.palette[car.label_id].tolist(), thickness=1, lineType=cv2.LINE_AA)
            if labels:
                img = detection.draw_labels(img, p1, p2, f'{car.type}:{car.ip}', car.label_id)

            gps_pixel = car_dict[car_id].gps_pixel
            img = cv2.circle(img, gps_pixel, 5, (0, 0, 255), 5)

            if (thresholding):
                left_top = [gps_pixel[0] - threshold, gps_pixel[1] - threshold]
                right_bottom = [gps_pixel[0] + threshold, gps_pixel[1] + threshold]
                left_top = clip_point(left_top, w, h)
                right_bottom = clip_point(right_bottom, w, h)
                
                if (not displayed):
                    cv2.rectangle(img, left_top, right_bottom, (0,255,0), thickness=3, lineType=cv2.LINE_AA)
                    img = cv2.circle(img, gps_pixel, 5, (0, 255, 0), 5)
                    displayed = True

            if (thresholding and masking):
                num_pixels += np.sum(mask[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]])
            elif (thresholding):
                num_pixels += (right_bottom[0] - left_top[0]) * (right_bottom[1] - left_top[1])
            else:
                num_pixels += h*w
            num_cars += 1

    data_size = num_pixels*2*8 # x,y with float32 (8 byte)
    print(data_size)
    print(num_cars)
    print(data_size/num_cars)

    return img



if __name__=='__main__':
    location_names = {
        "GW": "GwangGuo",
        "GS": "GyungSu",
        "KO": "Koorong",
        "NC": "NC",
        "SJ": "SeJong",
        "SG": "SoGong",
        "SB": "SungBock",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, help="location of CCTV image")
    parser.add_argument("--index", type=int, help="index of CCTV image")
    parser.add_argument("--compression", type=str, default="base", choices=["base", "rectangle_crop", "road_masking", "sidewalk_masking"], help="compression methods, 'base', 'rectangle_crop', 'road_masking' or 'sidewalk_masking'")
    parser.add_argument("--display", action="store_true", help="display the result")
    opt = parser.parse_args()

    ## config
    location = opt.location
    index = opt.index
    road_path = os.path.join(location_names[location], f'{location}00{index}')
    path_bundle = util.path_load(road_path)
    input_im, car_data, grid_data = util.load_input(path_bundle)

    threshold=0
    mask=np.zeros((input_im.shape[0], input_im.shape[1]))
    thresholding=False
    masking=False
    sidewalk=False

    if opt.compression == "sidewalk_masking":
        thresholding=True
        masking=True
        sidewalk=True
    elif opt.compression == "road_masking":
        thresholding=True
        masking=True
    elif opt.compression == "rectangle_crop":
        thresholding=True

    cls_model = YOLO('yolov8l.pt')

    res = cls_model(input_im)

    mapper = Vision2IP(input_im, grid_data, car_data)
    mapper.convert_gps2pixel()

    car_dict = mapper.match_coordinate(res)
    
    ## rectangle crop
    if thresholding:
        gps_scale = np.linalg.norm( mapper.gps_data[2] - mapper.gps_data[3] )
        pixel_scale = np.linalg.norm( mapper.road_square[2] - mapper.road_square[3] )
        threshold = pixel_scale / gps_scale / 5000

    ## road masking
    if masking:
        prefix = "_sidewalk" if sidewalk else ""
        mask = np.load(os.path.join("masking", f"{location}{prefix}_mask.npy"))

    ## plot
    res_img = plot(input_im, car_dict, threshold, mask, thresholding, masking)

    if masking:
        res_img = cv2.bitwise_and(res_img, res_img, mask=mask)

    if opt.display:
        cv2.imshow('img', res_img)
        cv2.waitKey(0)
        # cv2.imwrite(os.path.join('ip_img', road_path + '.jpg'), res_img)