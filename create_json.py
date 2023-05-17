import json
import os
import random

FILE_PATH =  os.path.join('json', 'GwangGuo', 'GW001.json')


class Car:
    def __init__(self, gps):
        self.ip = self.create_ip()
        self.gps = gps
        self.gps[0] += 37.
        self.gps[1] += 127.

    def create_ip(self):
        return '.'.join([str(random.randint(0, 256)) for i in range(4)])



if __name__ == '__main__':
    car_datas = {}

    cars = []

    cars.append(Car(gps=[.285331, .056075]))    
    cars.append(Car(gps=[.285334, .055901]))    
    cars.append(Car(gps=[.285212, .056000]))    
    cars.append(Car(gps=[.285106, .055836]))    
    cars.append(Car(gps=[.285182, .055781]))    
    cars.append(Car(gps=[.285255, .055682]))    
    cars.append(Car(gps=[.285287, .055636]))    
    cars.append(Car(gps=[.285293, .055680]))    
    cars.append(Car(gps=[.285301, .055726]))    
    cars.append(Car(gps=[.285322, .055700]))    
    cars.append(Car(gps=[.285343, .055677]))    
    cars.append(Car(gps=[.285310, .055630]))    
    cars.append(Car(gps=[.285428, .055662]))    
    cars.append(Car(gps=[.285375, .055875]))    
    cars.append(Car(gps=[.285422, .055931]))    
    cars.append(Car(gps=[.285465, .055830]))    
    cars.append(Car(gps=[.285587, .055504]))    

    result = dict(zip(range(len(cars)), [ob.__dict__ for ob in cars]))

    # json_string = json.dumps([ob.__dict__ for ob in cars])
    with open(FILE_PATH, 'w') as outfile:
        json.dump(result, outfile, indent=4)
    print("Saved {} cars at {}".format(len(cars), FILE_PATH))