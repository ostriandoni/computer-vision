import cv2
image1 = cv2.imread('gs1.png')
ret, thresh1 = cv2.threshold(image1, 120, 255, cv2.THRESH_BINARY)
print(ret)
print(thresh1)