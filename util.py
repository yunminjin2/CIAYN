import cv2
import os
import json


def load_input(path_bundle):
    im = cv2.imread(path_bundle['img_path'])  # OpenCV
    with open(path_bundle['json_path'], "r") as f:
        car_data = json.load(f)
    with open(path_bundle['grid_path'], "r") as f:
        grid_data = json.load(f)[path_bundle['road_seperator']]

    return im, car_data, grid_data


def path_load(road_path):
    road_seperator = road_path[-5:-3]
    img_path = os.path.join('img', road_path + '.jpg')
    json_path = os.path.join('json', road_path + '.json')
    grid_path = 'grid_info.json'
    return {'img_path': img_path, 'json_path': json_path, 'grid_path': grid_path, 'road_seperator': road_seperator}