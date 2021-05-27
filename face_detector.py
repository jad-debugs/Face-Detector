import cv2

# loading trained images thanks to opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load in image that is detected
img = cv2.imread('isayama.jpg')

# converted to gray scale so easier on machine
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces, guves us top left and off set from that for right point
face_coor = trained_face_data.detectMultiScale(grayscale_img)

# draw rectangle around face
(x, y, w, h) = face_coor[0]
# coor of top left -> add x1 x2, y1 y2 -> color -> thickness
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

print(face_coor)

# showing image, first param is name of window
cv2.imshow('Face detector', img)

# this will wait until a key is pressed (so you can view image)
cv2.waitKey()



print("Code completed")
