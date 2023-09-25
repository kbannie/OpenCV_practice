import cv2
import numpy as np

model_name='./Data/res10_300x300_ssd_iter_140000.caffemodel'
prototext_name='./Data/deploy.prototxt.txt' #아키텍처 구성도
min_confidence=0.5 #50% 이상 확률만 보여주기
file_name='./Data/image/marathon_03.jpg'

def detectAndDisplay(frame):
    #blob형태로 변화하기
    #dnn중 Caffe가 간단해서 자주 사용 
    model=cv2.dnn.readNetFromCaffe(prototext_name, model_name)

    #Resizing to a fixed 300x300 pixels and then normalizing it
    blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0,(300,300),(104.0, 177.0,123.0))
    model.setInput(blob)
    detections=model.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]): #3번째 내용까지 loop 돌리기
            confidence = detections[0, 0, i, 2] #확률값

            if confidence > min_confidence: #0.5보다 큰 경우에는 박스 그려주기
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    print(confidence, startX, startY, endX, endY)

                    # draw the bounding box of the face along with the associated
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10 #text 위치 지정
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # show the output image
    cv2.imshow("Face Detection by dnn", frame)
    

img=cv2.imread(file_name)
print("width:{}pixels".format(img.shape[1]))
print("height:{}pixels".format(img.shape[0]))
print("channels:{}pixels".format(img.shape[2]))

(height, width)=img.shape[:2]

cv2.imshow("Original Image", img)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()