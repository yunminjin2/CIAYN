import json
from ultralytics import YOLO

import os
import numpy as np
import cv2

import util
import detection

class GPS2Pixel:
    def __init__(self, grid_data):
        self.grid_data = grid_data
        self.h, self.w = grid_data['length'][0], grid_data['length'][1]

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
            [3/8*self.h, 3/8*self.w],
            [5/8*self.h, 3/8*self.w], 
            [5/8*self.h, 5/8*self.w], 
            [3/8*self.h, 5/8*self.w]]
        )

        self.img2grid_map = cv2.getPerspectiveTransform(road_square, trg)
        self.grid2img_map = cv2.getPerspectiveTransform(trg, road_square)

class Vision2IP(GPS2Pixel):
    def __init__(self, grid_data):
        super().__init__(grid_data)

    def convert_gps2pixel(self, car_data):
        trans_car_data = car_data.copy()
        for car_id in car_data.keys():
            pixel_coor = np.array(self.gps2img(car_data[car_id]['gps']), dtype=np.uint32)
            trans_car_data[car_id]['pixel'] = pixel_coor.tolist()

        return trans_car_data

    def match_coordinate(self, res):
        None
    def match_ip():
        None

if __name__=='__main__':
    road_path = os.path.join('Koorong', 'KO001')
    path_bundle = util.path_load(road_path)
    input_im, car_data, grid_data = util.load_input(path_bundle)

    cls_model = YOLO('yolov8l.pt')

    res = cls_model(input_im)

    mapper = Vision2IP(grid_data)
    car_data = mapper.convert_gps2pixel(car_data)


    img = detection.plot(input_im, res)
    # import pdb; pdb.set_trace()
    for car_id in car_data.keys():
        print( car_data[car_id]['pixel'])
        img = cv2.circle(img, car_data[car_id]['pixel'], 5, (0, 0, 255), 5)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)

    cv2.imwrite(os.path.join('gps_img', road_path + '.jpg'), img)