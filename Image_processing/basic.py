import cv2

print("OpenCV version:")
print(cv2.__version__)

img=cv2.imread("./photo/kkk.png")
print("width: {} pixels".format(img.shape[1])) 
print("height: {} pixels".format(img.shape[0]))
print("channels: {} pixels".format(img.shape[2])) #RGB

cv2.imshow("djkd",img)

(b,g,r)=img[0,0]
print("Pixel at (0,0)-Red:{},Green:{},Blue:{}".format(r,g,b))


dot=img[50:100, 50:100]
# cv2.imshow("dot",dot)

img[50:100, 50:100]=(0,0,255) #BGR

cv2.rectangle(img,(150,50),(200,100),(0,255,0),5) #좌측하단, 우측상단, 색상, 선의굵기
cv2.circle(img,(275,75),25,(0,255,255),-1) #중간위치, 반지름크기, 색상, -1=전체채우기
cv2.line(img,(350,100),(400,100),(255,0,0),5)
cv2.putText(img,'kabeen',(100,200),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),4) 
#좌측하단, 폰트모양, 폰트크기, 색상, 굵기

cv2.imshow("draw",img)

cv2.waitKey(0)
cv2.destroyAllWindows()