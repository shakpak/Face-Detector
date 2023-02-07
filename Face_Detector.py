import cv2


#Import pretrained dataset
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Import the Image to Detect face(s) in
img = cv2.imread('facefront.jpeg')

#covert to gray scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Face(s)
face_coordinate = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinate)
(x,y,w,h) = face_coordinate[0]
cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Clever face detector', img)
cv2.waitKey()
