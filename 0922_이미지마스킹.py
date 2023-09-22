import cv2
import numpy as np

print("OpenCV version:")
print(cv2.__version__)

img=cv2.imread("./photo/cat.png")
print("width: {} pixels".format(img.shape[1])) 
print("height: {} pixels".format(img.shape[0]))
print("channels: {} pixels".format(img.shape[2]))

(height,width)=img.shape[:2] #shape의[0]값을 height에, [1]의 값을 width에 넣기
center=(width//2,height//2) #integer값으로 

cv2.imshow("image",img)

mask=np.zeros(img.shape[:2],dtype="uint8") #모든 영역에 0을 넣음

cv2.circle(mask, center, 200, (255,255,255),-1) # 300:원의크기, 색상, 두께

cv2.imshow("mask",mask)

masked=cv2.bitwise_and(img, img, mask=mask) #bitwise_and:공통적인 영역
cv2.imshow("image with mask", masked)

cv2.waitKey(0)
cv2.destroyAllWindows()