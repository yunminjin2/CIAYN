import os.path as osp
import cv2
import numpy as np

target = "SJ_sidewalk_mask"

img = cv2.imread(osp.join('masking', f'{target}.png'), cv2.IMREAD_GRAYSCALE)

mask = (img==255)
mask = np.uint8(mask)

# result = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow('img', result)
# cv2.waitKey(0)

np.save(osp.join('masking', f'{target}.npy'), mask)