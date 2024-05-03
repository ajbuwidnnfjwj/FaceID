import cv2 as cv
import numpy as np
import ImageProcess

sumin = cv.imread("./FaceID/images/sumin.jpg")
isa = cv.imread("./FaceID/images/isa_1.jpg")

face_s = ImageProcess.getFacePart(sumin)
face_i = ImageProcess.getFacePart(isa)

image_with_key_points_sumin = cv.drawKeypoints(face_s, ImageProcess.keyPoints(face_s)[0], None)
image_with_key_points_isa = cv.drawKeypoints(face_i, ImageProcess.keyPoints(face_i)[0], None)

cv.imshow("sumin", image_with_key_points_sumin)
cv.imshow("isa", image_with_key_points_isa)
cv.waitKey()