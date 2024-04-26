import cv2 as cv

image = cv.imread("./images.jpg")

resized = cv.resize(image, dsize = (0, 0), fx=0.2, fy=0.2, interpolation=cv.INTER_LINEAR)
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 이미지에서 얼굴을 검출합니다.
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 검출된 얼굴 주위에 사각형을 그립니다.
# for (x, y, w, h) in faces:
#     cv.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
x,y,w,h = faces[0]
face_img = resized[x:x+w, y:y+h, :]

# 결과 이미지를 화면에 출력합니다.
cv.imshow('Face Detection', face_img)
cv.waitKey(0)
cv.destroyAllWindows()