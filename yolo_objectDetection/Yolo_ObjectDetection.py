import cv2
import numpy as np

#Load Yolo
net=cv2.dnn.readNet("./Data/yolo/yolov3.weights", "./Data/yolo/yolov3.cfg") #yolo3.weights:이미 학습된 모델
classes=[]
with open("./Data/yolo/coco.names","r") as f: #파일 갖고오기
    classes=[line.strip() for line in f.readlines()] #classes 배열에 80개가 모두 들어감
layer_names=net.getLayerNames() 
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # yolo 작동하는 방식
colors=np.random.uniform(0,255,size=(len(classes),3)) #0부터 255까지 size만큼 채널수는 3으로 해서 랜덤으로 색상지정

#Loading image
img=cv2.imread("./Data/image/yolo_01.jpg")
img=cv2.resize(img, None, fx=0.4, fy=0.4) #0.4만큼 줄이기
height, width, channels=img.shape

#Detecting objects
#blob타입으로 갖고오기
#yolo는 3가지 크기의 데이터를 표준화함
#(320,320) : 작지만 정확도 떨어짐, 속도 빠름
#(416,416) : 중간
#(609,609) : 정확도 높음, 속도 느림
blob=cv2.dnn.blobFromImage(img, 0.00392, (416,416),(0,0,0),True, crop=False)
net.setInput(blob)
outs=net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
