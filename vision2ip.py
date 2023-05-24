import json
from ultralytics import YOLO

import os
import numpy as np

import detection
import util





if __name__=='__main__':
    road_path = os.path.join('GwangGuo', 'GW001')
    path_bundle = util.path_load(road_path)
    input_im, car_data, grid_data = util.load_input(path_bundle)

    cls_model = YOLO('yolov8l.pt')

    res = cls_model(input_im)


    import pdb; pdb.set_trace()