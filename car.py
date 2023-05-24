class IPCar:
    def __init__(self, ip, label_id, bbox, gps, gps_pixel, label):
        self.ip = ip
        self.bbox = bbox
        self.gps = gps
        self.gps_pixel = gps_pixel
        self.label_id = label_id
        self.type = label

        
