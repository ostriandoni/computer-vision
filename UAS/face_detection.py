import cv2
import sys

# mendeteksi parameter input berupa image yang akan diproses
imagePath = sys.argv[1]

# memuat konfigurasi object detection
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# membaca image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# melakukan deteksi wajah dari image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (25, 25)
)

print("Jumlah wajah yang terdeteksi adalah {0}".format(len(faces)))

# Menggambarkan persegi pada wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces Recognition", image)
cv2.waitKey(0)
