import cv2 as cv
import numpy as np


FACE_CASCADE = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detectFace(image, return_faces = True) -> tuple:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if return_faces:
        return faces
    else:
        return True if len(faces) == 1 else False
    
def getFacePart(image) -> np.array:
    # returns only face in image
    assert image is not None, 'cannot load image'
    faces = FACE_CASCADE.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    x,y,w,h = faces[0]
    face_img = image[x:x+w, y:y+h, :]
    return face_img