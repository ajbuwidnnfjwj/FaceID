import cv2 as cv
import numpy as np
import ImageProcess

isa_1 = ImageProcess.getFacePart(cv.imread("./FaceID/images/isa_1.jpg"))
isa = ImageProcess.getFacePart(cv.imread("./FaceID/images/isa_2.jpg"))

keypoints_s, descriptors_s = ImageProcess.keyPoints(isa_1)
keypoints_i, descriptors_i = ImageProcess.keyPoints(isa)

matcher = cv.BFMatcher()
matches = matcher.knnMatch(descriptors_s, descriptors_i, k = 2)

good_match = sorted([m for m, _ in matches], key = lambda x : x.distance)[:100]
print(np.array(good_match))

points1 = [keypoints_s[m.queryIdx].pt for m in good_match]
points2 = [keypoints_i[m.trainIdx].pt for m in good_match]

image_matches = cv.drawMatches(isa_1, keypoints_s, isa, keypoints_i, good_match, None)

# 결과 이미지를 화면에 표시합니다.
cv.imshow('Correspondences', image_matches)
cv.waitKey(0)
cv.destroyAllWindows()