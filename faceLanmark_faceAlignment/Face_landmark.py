import numpy as np
import dlib
import cv2

#landmark: 68개 점으로 얼굴 중요 포인트 나타내기
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = 'C:/OpenSK/ai_cv/model/shape_predictor_68_face_landmarks.dat'
image_file = 'C:/OpenSK/ai_cv/image/marathon_03.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #사진을 단순화해서 인식률 높이기

rects = detector(gray, 1)
print("Number of faces detected: {}".format(len(rects)))


for (i, rect) in enumerate(rects):
        points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
        show_parts = points[ALL] #68개의 좌표 / point[점으로 나타낼 부분 선택]
        for (i, point) in enumerate(show_parts):
                x = point[0,0]
                y = point[0,1]
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
                cv2.putText(image, "{}".format(i + 1), (x, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)   
