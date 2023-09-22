import cv2
import numpy as np

print("OpenCV version:")
print(cv2.__version__)

img=cv2.imread("./photo/kkk.png")

(height,width)=img.shape[:2] #shape의[0]값을 height에, [1]의 값을 width에 넣기
center=(width//2,height//2) #integer값으로 

cv2.imshow("image",img)

#이미지 위치 바꾸기
move=np.float32([[1,0,-100],[0,1,100]]) #[1,0,100]:위아래로/[0,1,100]:좌우로 //세번째값이 양수는 down&right, 음수는 up&left
moved=cv2.warpAffine(img,move,(width,height)) #warpAffine(옮길대상, 움직이는 좌표값,(움직이는 크기))
cv2.imshow("Moved down:+, up: - and right : +, left -",moved)

#이미지 돌리기
move2=cv2.getRotationMatrix2D(center,90,1.0) #center, 각도, 크기(scale)
rotated=cv2.warpAffine(img,move2,(width,height)) 
cv2.imshow("Rotated clockwise degrees",rotated)

#이미지 사이즈 줄이기
ratio=200.0/width
dimension=(200,int(height*ratio))
resized=cv2.resize(img,dimension, interpolation=cv2.INTER_AREA) #INTER_AREA:영역보관법? (보통 사용)
cv2.imshow("Resized",resized)

#이미지 좌우, 상하 대칭하기
flipped=cv2.flip(img,1)
cv2.imshow("Flipped Horozontal 1, Vertical 0, both -1",flipped)

cv2.waitKey(0)
cv2.destroyAllWindows()