import cv2

# loading trained images thanks to opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load in image that is detected
img = cv2.imread('isayama.jpg')

# converted to gray scale so easier on machine
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# showing image, first param is name of window
cv2.imshow('Face detector', grayscale_img)

# this will wait until a key is pressed (so you can view image)
cv2.waitKey()



print("Code completed")
