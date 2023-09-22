import cv2
import numpy as np

def detectAndDisplay(frame):
    #채널이 여러개면 변수가 많아지며 정확도가 떨어지니 gray로 만들기
    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray=cv2.equalizeHist(frame_gray) #히스토그램을 사용하여 단순화하여 인식을 편하게 함
    #--Detect faces
    faces=face_cascade.detectMultiScale(frame_gray) #쪼개진 영역을 하나로
    for(x,y,w,h) in faces: #x축, y축, 넓이, 높이
        center=(x+w//2, y+h//2) #중간위치
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        faceROI=frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes=eyes_cascade.detectMultiScale(faceROI)
        for(x2,y2,w2,h2) in eyes:
            eye_center=(x+x2+w2//2, y+y2+h2//2)
            radius=int(round((w2+h2)*0.25)) 
            frame=cv2.circle(frame, eye_center, radius, (255,0,0),4)
        cv2.imshow("capture-face detection",frame)

print("OpenCV")
print(cv2.__version__)

img=cv2.imread("./ai_cv/image/marathon_02.jpg")
print("width:{} pixels".format(img.shape[1]))
print("height:{} pixels".format(img.shape[0]))
print("channels:{} pixels".format(img.shape[2]))

(height, width)=img.shape[:2]

#cv2.imshow("original",img)

#xml을 이용한 얼굴, 눈 인식
face_cascade_name='./ai_cv/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name='./ai_cv/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade=cv2.CascadeClassifier()
eyes_cascade=cv2.CascadeClassifier()

#-- 1.Load the cascaeds
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("error")
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print("error")
    exit(0)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()