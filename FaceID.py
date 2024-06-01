import Classifier
import ImageProcess
import cv2 as cv

def getUserFace():
    webcam = cv.VideoCapture(0)
    FACE_CASCADE = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while webcam.isOpened():
        status, frame = webcam.read()                 

        #프레임에서 얼굴을 찾고 얼굴 주위에 사각형을 그려 표시
        faces = FACE_CASCADE.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        copy_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv.rectangle(copy_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        #프레임 표시
        if status:
            cv.imshow("FaceID - Get Face", copy_frame) 

        #강제 종료 프로세스
        key = cv.waitKey(10)
        if key == 27:
            break
        if cv.getWindowProperty("FaceID - Get Face", cv.WND_PROP_VISIBLE) < 1:
            break
        

if __name__ == '__main__':
    print('type 1 to register new face')
    print('type 2 to run FaceID')
    option = int(input('Enter integer: '))

    if option == 1:
        getUserFace()
        cv.destroyAllWindows()
    elif option == 2:
        pass
    else:
        print('wrong input')