import cv2 as cv
import numpy as np


FACE_CASCADE = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detectFace(image) -> tuple:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return True if len(faces) == 1 else False
    
def getFacePart(image) -> np.array:
    # returns only face in image
    assert image is not None, 'cannot load image'
    faces = FACE_CASCADE.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    x,y,w,h = faces[0]
    face_img = image[x:x+w, y:y+h, :]
    return face_img

if __name__ == '__main__':
    import os
    path = '.\FaceID\images\\isa'
    files = os.listdir(path)

    for i, file_name in enumerate(files):
        image_name = os.path.join(path, file_name)
        image = cv.imread(image_name)
        if detectFace(image) == False:
            os.remove(image_name)
        else:
            image = getFacePart(image)
            cv.imwrite(image_name, image)
