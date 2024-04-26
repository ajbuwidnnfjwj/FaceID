import cv2 as cv

src = cv.imread("apple_books.jpg")
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
target = cv.imread("apple.jpg", cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create(
    nfeatures=40000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
)

kp1, des1 = orb.detectAndCompute(gray, None)
kp2, des2 = orb.detectAndCompute(target, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

for i in matches[:100]:
    idx = i.queryIdx
    x1, y1 = kp1[idx].pt
    cv.circle(src, (int(x1), int(y1)), 3, (255, 0, 0), 3)

cv.imshow("src", src)
cv.waitKey()