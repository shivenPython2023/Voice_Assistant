import cv2
from picamera2 import Picamera2

# Initialize PiCamera2
#0 should be right side
picam1 = Picamera2(0)
picam2 = Picamera2(1) 

picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

picam1.configure(picam1.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam1.start()                                                                                                                                                     


num = 0


while True:

    img = picam2.capture_array()  # Capture an image frame from camera 0
    img2 = picam1.capture_array()  # Capture an image frame from camera 1


    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)