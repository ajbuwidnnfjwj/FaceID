from Classifier import Classifier
import ImageProcess
import cv2 as cv

def getUserFace():
    webcam = cv.VideoCapture(0)

    while webcam.isOpened():
        status, frame = webcam.read()                 

        #프레임에서 얼굴을 찾고 얼굴 주위에 사각형을 그려 표시
        copy_frame = frame.copy()
        faces = ImageProcess.detectFace(copy_frame)
        for (x, y, w, h) in faces:
            cv.rectangle(copy_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        #프레임 표시
        if status:
            cv.imshow("FaceID - Get Face", copy_frame) 

        #얼굴을 찾으면 창을 닫음
        if len(faces) == 1:
            return frame
        
        #강제 종료 프로세스
        key = cv.waitKey(10)
        if key == 27:
            break
        if cv.getWindowProperty("FaceID - Get Face", cv.WND_PROP_VISIBLE) < 1:
            break
        
if __name__ == '__main__':
    while True:
        print('type 1 to run FaceID')
        print('type 2 to register new face')
        option = int(input('Enter integer: '))

        if option == 1:
            face_img = getUserFace()
            cv.destroyAllWindows()
            classifier = Classifier()
            classifier.classifyFace(face_img)
        elif option == 2:
            import os
            path = '.\FaceID\images\\0'
            files = os.listdir(path)
            for i, file_name in enumerate(files):
                image_name = os.path.join(path, file_name)
                image = cv.imread(image_name)
                if ImageProcess.detectFace(image, False):
                    image = ImageProcess.getFacePart(image)
                    cv.imwrite(image_name, image)

            classifier = Classifier()
            classifier.trainAndSaveModel()
        else:
            print('wrong input')
            break