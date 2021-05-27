import cv2

# loading trained data thanks to opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load in image that is detected
# img = cv2.imread('isayama.jpg')
# captures data from default cam
webcam = cv2.VideoCapture(0)

# iterate over all framers until stopped
while True:
    # read current frame
    successful_frame_read, frame = webcam.read()
    # convert to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get coor of face    
    face_coor = trained_face_data.detectMultiScale(grayscale_img)

    for (x, y, w, h) in face_coor:
        # draw rectangle around face
        # coor of top left -> add x1 x2, y1 y2 -> color -> thickness
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 

    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)

    # if q or Q is tapped, then quit
    if key == 81 or key == 113:
        break

webcam.release()
