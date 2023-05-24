import cv2
import os
import json
import numpy as np


palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                    dtype=np.uint8)

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