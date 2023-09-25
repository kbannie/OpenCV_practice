import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

model_name='./Data/res10_300x300_ssd_iter_140000.caffemodel'
prototext_name='./Data/deploy.prototxt.txt'
min_confidence = 0.3
file_name = "./Data/image/marathon_01.jpg"
title_name = 'dnn Deep Learnig object detection'
frame_width = 300
frame_height = 300

def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./image",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('File name : ', file_name)
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    (height, width) = read_image.shape[:2]
    detectAndDisplay(read_image, width, height)

def detectAndDisplay(frame, w,h):
    #blob형태로 변화하기
    #dnn중 Caffe가 간단해서 자주 사용 
    model=cv2.dnn.readNetFromCaffe(prototext_name, model_name)

    #Resizing to a fixed 300x300 pixels and then normalizing it
    blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0,(300,300),(104.0, 177.0,123.0))
    model.setInput(blob)
    detections=model.forward()
    min_confidence=float(sizeSpin.get())
    # loop over the detections
    for i in range(0, detections.shape[2]): #3번째 내용까지 loop 돌리기
            confidence = detections[0, 0, i, 2] #확률값

            if confidence > min_confidence: #0.5보다 큰 경우에는 박스 그려주기
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
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
    #cv2.imshow("Face Detection by dnn", frame)
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image=Image.fromarray(image)
    imgtk=ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    detection.image=imgtk

#main
main=Tk()
main.title(title_name)
main.geometry()

read_image=cv2.imread(file_name)
image=cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
image=Image.fromarray(image)
imgtk=ImageTk.PhotoImage(image=image)
(height, width)=read_image.shape[:2]

label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)
sizeLabel=Label(main, text='Min Confidence : ')                
sizeLabel.grid(row=1,column=0)
sizeVal  = IntVar(value=min_confidence)
sizeSpin = Spinbox(main, textvariable=sizeVal,from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin.grid(row=1, column=1)
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(W, E))
detection=Label(main, image=imgtk)
detection.grid(row=2,column=0,columnspan=4)
detectAndDisplay(read_image, width, height)

main.mainloop()
