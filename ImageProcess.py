import cv2 as cv
import numpy as np

FACE_CASCADE = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detectFace(image) -> tuple:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    assert len(faces) == 1, 'no face or too many faces found'
    return faces
    
def getFacePart(image) -> np.array:
    # returns only face in image
    assert image is not None, 'cannot load image'
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    x,y,w,h = faces[0]
    face_img = image[x:x+w, y:y+h, :]
    return face_img

def keyPoints(image, 
    method = cv.ORB_create(
        nfeatures=40000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )) -> tuple:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return method.detectAndCompute(gray, None)

if __name__ == '__main__':
    image = cv.imread('./FaceID/images/isa_1.jpg')
    print(image.shape())
    resize = cv.resize(image, dsize = (0, 0), fx=0.4, fy=0.4, interpolation=cv.INTER_LINEAR)
    image = getFacePart(resize)
    image_with_key_points = cv.drawKeypoints(image, keyPoints(image)[0], None)
    cv.imshow('Image with Keypoints', image_with_key_points)
    cv.waitKey(0)
    cv.destroyAllWindows()