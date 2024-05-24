import cv2 as cv
import numpy as np
import ImageProcess
import os

path = '.\FaceID\images\\isa'
files = os.listdir(path)

for file_name in files:
    image_name = os.path.join(path, file_name)
    image = cv.imread(image_name)
    # image = ImageProcess.getFacePart(image)
    cv.imwrite(image_name, image)