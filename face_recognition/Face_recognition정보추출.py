#recognition 정보가 있어야 하니 10장-25장 준비해서 특성 추출하기
#이미지 선처리가 중요하나 그것은 안함 (face Landmark / face Alignment)
# 128d 128vector 이미지를 encoding

import cv2
import face_recognition
import pickle #정보를 serialize(일렬로 세우기)한 후 다른 형태로 deserealize해줌

dataset_paths=['./Data/dataset/jang/']
names=['Jang']
number_images=10
image_type='.png'
encoding_file='encodings3.pickle'
model_method='cnn' #?네? 대표적인 시각화 방법으로 정확하지만 느림/ hog는 빠르지만 정확도 낮음

knownEncodings=[]
knownNames=[]

for (i, dataset_path) in enumerate(dataset_paths): #enumerate : 인덱스와 원소를 이루어진 튜플로 만들어줌
    #이름 추출
    name=names[i]

    for idx in range(number_images): 
        file_name=dataset_path+str(idx+1)+image_type #이미지 읽어오기
        image=cv2.imread(file_name)
        rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #boundary box
        #얼굴을 인식해서 위치를 찾음 cnn방식으로
        boxes=face_recognition.face_locations(rgb, model=model_method)

        #encoding
        encodings=face_recognition.face_encodings(rgb, boxes) #얼굴만 갖고옴
        # 128개의 real number로 되어있으니 loop를 돌리기
        for encoding in encodings:
            print(file_name, name, encoding)
            knownEncodings.append(encoding) 
            knownNames.append(name)

#save the facial encoding + 파일로 저장하기
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encoding_file, "wb") #파일 열기
f.write(pickle.dumps(data)) #데이터 넣어주기
f.close()

