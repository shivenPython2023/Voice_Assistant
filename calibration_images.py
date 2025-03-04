import cv2

for i in range(10):
    cap_test = cv2.VideoCapture(i)
    if cap_test.isOpened():
        print(f"Camera index {i} is available.")
        cap_test.release()

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap2 = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

# Set resolution to 640x480 for both cameras
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
num = 0


while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()


    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
        cv2.imwrite('images/stereoright/imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)
