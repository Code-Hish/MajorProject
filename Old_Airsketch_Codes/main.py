import cv2
import os
import time
import handtracking as htm
import numpy as np

# Import the images stored in header folder
images = "header"
mylist = os.listdir(images)
print(mylist)
list = []

# Store the images in a list
for i in mylist:
    img = cv2.imread(f'{images}/{i}')
    list.append(img)

print(len(list))

h = list[0]
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionCon=0.85)
color = (255, 0, 0)
x_prev = 0
y_prev = 0
# Create a canvas where the outputs will  be displayed,set its background color to white
canvas = np.zeros((480, 640, 3), np.uint8)
canvas[:, :, :] = (255, 255, 255)
# Set the thickness of brush
thick = 10


drawing = False
# We need to overlay the other image...
while True:
    # Import image
    succ, im = cap.read()
    # Flip the image so that its easier to draw
    im = cv2.flip(im, 1)

    # Find the hand landmarks using handtracking module
    im = detector.find_hands(im)
    # Getthe positions of the landmarks
    lmlist = detector.find_pos(im, draw=False)

    if (len(lmlist) != 0):
        # print(lmlist)
        # Tip of the index finger (grab only the x and y coordinates)
        x1, y1 = lmlist[8][1:]
        # Tip of the middle finger (grab only the x and y coordinates)
        x2, y2 = lmlist[12][1:]

        # Use the predefined method fingers_up to check the fingers which are up
        fingers = detector.fingers_up()
        fingers = detector.fingers_up()

        # 🧹 Clear canvas when all fingers are up
        if all(fingers):
            canvas = np.zeros((480, 640, 3), np.uint8)
            canvas[:, :, :] = (255, 255, 255)
        # print(fingers)
        # Now check which mode to work in.. Draw or selection
        elif (fingers[1] and fingers[2]):
            print("Selection mode")
            x_prev = 0
            y_prev = 0
            drawing = False
            # If we are the top of image (we need to change the display of painter)
            if y1 < 62:
                if 90 < x1 < 150:
                    thick = 10
                    color = (255, 0, 0)  # Brush0 is of blue color
                    h = list[0]
                elif 150 < x1 < 235:
                    thick = 10
                    color = (0, 165, 255)    # Brush1 is of orange color
                    h = list[1]
                elif 235 < x1 < 305:
                    thick = 10
                    color = (128, 0, 128)    # Brush2 is of violet color
                    h = list[2]
                elif 330 < x1 < 400:
                    thick = 10
                    color = (0, 0, 255)      # Brush3 is of red color
                    h = list[3]
                elif 450 < x1 < 510:
                    thick = 80
                    color = (255, 255, 255)  # for eraser
                    h = list[4]
            cv2.rectangle(im, (x1, y1-20), (x2, y2+20),
                          color, cv2.FILLED)
        elif (fingers[1]):
            cv2.circle(im, (x1, y1), 7, color, cv2.FILLED)
            if not drawing:
                x_prev, y_prev = x1, y1
                drawing = True
                continue
            
            cv2.line(im, (x_prev, y_prev), (x1, y1), color, thick)
            cv2.line(canvas, (x_prev, y_prev), (x1, y1), color, thick)
            x_prev, y_prev = x1, y1
            
            # The above values will keep on updating, so print the drawing on a canvas
            print("Drawing mode")
        else:
            drawing = False
            x_prev = 0
            y_prev = 0

    # When the index finger is up, then only draw
    # If both fingers up,only allow selection of paint brush
    # Webcam resolution is 640*480..and we need the image to be overlaid at the top
    im[0:62, 0:640] = h
    # Blends the canvas with image
    im = cv2.addWeighted(im, 0.5, canvas, 0.5, 0)
    cv2.imshow("Image", im)
    cv2.imshow("Image Canvas", canvas)
    cv2.waitKey(1)
