import json
import os
import random

FILE_PATH =  os.path.join('json', 'Koorong', 'KO001.json')


class Car:
    def __init__(self, gps):
        self.ip = self.create_ip()
        self.gps = gps
        self.gps[0] += 37.
        self.gps[1] += 126.

    def create_ip(self):
        return '.'.join([str(random.randint(0, 256)) for i in range(4)])



if __name__ == '__main__':
    car_datas = {}

    cars = []

    cars.append(Car(gps=[.423727, .993252]))    
    cars.append(Car(gps=[.423954, .993203]))    
    cars.append(Car(gps=[.423944, .993237]))    
    cars.append(Car(gps=[.423870, .993272]))    
    cars.append(Car(gps=[.423957, .993251]))    
    cars.append(Car(gps=[.423907, .993341]))    
    cars.append(Car(gps=[.423935, .993372]))    





    result = dict(zip(range(len(cars)), [ob.__dict__ for ob in cars]))

    # json_string = json.dumps([ob.__dict__ for ob in cars])
    with open(FILE_PATH, 'w') as outfile:
        json.dump(result, outfile, indent=4)
    print("Saved {} cars at {}".format(len(cars), FILE_PATH))