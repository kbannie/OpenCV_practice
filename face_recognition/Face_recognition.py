import cv2
import face_recognition
import pickle
import time #시간 측정


image_file="C:/OpenSK/Data/image/yu.png"
encoding_file="C:/OpenSK/encodings.pickle"
unknown_name="Unknown"
model_method="cnn"

def detectAndDisplay(image):
    start_time=time.time()
    rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #얼굴 인식 부분 박스 그리기
    boxes = face_recognition.face_locations(rgb,
        model=model_method)
    #박스 인식 부분을 encoding 해주기
    encodings = face_recognition.face_encodings(rgb, boxes)

    names=[]

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = unknown_name

        #매치 여부 확인
        if True in matches:
            #매치 index 값 주기
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            #어떤 내용으로 매치 되었는지 for문 돌리기
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            #가장 많이 이름이 나온 것으로 결정
            name = max(counts, key=counts.get)
        
        #이름 넣어주기/없으면 unknown
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        #인식된 이름을 이미지 위에 넣어주기
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if(name == unknown_name): #unkonw일 경우 다르게 설정
            color = (0, 0, 255)
            line = 1
            name = ''
            
        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, line)

    #시간측정
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    # show the output image
    cv2.imshow("Recognition", image)
    


data=pickle.loads(open(encoding_file,"rb").read())

image=cv2.imread(image_file)
detectAndDisplay(image)
cv2.waitKey(0)
cv2.destroyAllWindows()