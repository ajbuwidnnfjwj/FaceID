import cv2 as cv
import numpy as np
import ImageProcess
import os

#이미지 전처리 - 얼굴을 찾아 얼굴부분만 저장
path = '.\FaceID\images\\isa'
files = os.listdir(path)

for i, file_name in enumerate(files):
    image_name = os.path.join(path, file_name)
    image = cv.imread(image_name)
    if ImageProcess.detectFace(image) == False:
        os.remove(image_name)
    else:
        image = ImageProcess.getFacePart(image)
        cv.imwrite(image_name, image)

