import cv2
import mediapipe as mp
import time


class handDetector:
    # add one extra parameter model complexity as the parameter
    # Initialization
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComp = modelComp
        # using the hands module of mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon,
                                        model_complexity=self.modelComp)
        self.mpDraw = mp.solutions.drawing_utils

        # For the 5 fingers initialise their landmark positions
        self.tip = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        # Send the RGB image to the object
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # to detect the hand landmarks and establish the connections
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_pos(self, img, handNo=0, draw=True):
        # to return a list of landmarks
        self.lmList = []
        if self.results.multi_hand_landmarks:
            # For a particular hand
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                # prints in form of x,y,z coordinates, we need t use these x and y coords to locate the landmarks and use them
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    # Reduced circle size and changed color to yellow
                    cv2.circle(img, (cx, cy), 7, (0, 255, 255), cv2.FILLED)
        return self.lmList

    # To check which of the fingers are up
    def fingers_up(self):
        list = []
        # For thumb
        if (self.lmList[self.tip[0]][1] < self.lmList[self.tip[0]-1][1]):
            list.append(1)
        else:
            list.append(0)

        # For the other four fingers (1-4)
        for i in range(1, 5):
            if self.lmList[self.tip[i]][2] < self.lmList[self.tip[i]-2][2]:
                list.append(1)
            else:
                list.append(0)
        return list


def main():
    # To generate the fps(frame per second), initialise two variable previous and current time
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmlist = detector.find_pos(img)
        if len(lmlist) != 0:
            print(lmlist[4])
        # Current time
        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        # print the fps
        cv2.putText(img, str(int(fps)), (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
