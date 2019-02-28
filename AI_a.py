import os,os.path
import numpy as np
import cv2

import torch
from torchvision import transforms

image_sort = ['png','jpg','jpeg']

image_list = []

path = './data'

for filename in os.listdir(path):
    print(filename)
    for i in image_sort:
        print(i)

        if filename.lower().endswith(i):


            filename = path + '/' + filename
            print(filename)

            image = cv2.imread(filename)

            image_list.append(image)


            re_image = cv2.resize(image,(100,100))

            image_list.append(re_image)

image_arr = np.asarray(image_list)

img_as_tensor = []

for i in range(len(image_arr)):

    img_as_tensor.append(transforms.ToTensor()(image_arr[i]))


print(img_as_tensor)