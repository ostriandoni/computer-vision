import cv2

# membaca gambar dari local file
img = cv2.imread('snake.jpg') 
# menampilkan gambar original
cv2.imshow('Original', img)
cv2.waitKey(0)

# mengkonversi gambar menjadi graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# melakukan blur terhadap gambar untuk hasil deteksi edge yang lebih baik
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

# melakukan Sobel Edge Detection pada sumbu X
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
# melakukan Sobel Edge Detection pada sumbu Y
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
# mengkombinasikan Sobel Edge Detection pada sumbu X dan Y
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

# menampilkan gambar hasil Sobel Edge Detection
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
# menampilkan gambar hasil Canny Edge Detection
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
