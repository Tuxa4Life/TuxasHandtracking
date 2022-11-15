import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMs, self.mpHands.HAND_CONNECTIONS)
        
        return img

    
    def findPosition(self, img, handId=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            targetHand = self.results.multi_hand_landmarks[handId]
            for id, lm in enumerate(targetHand.landmark):
                h, w, c = img.shape
                cx , cy = int(lm.x * w), int(lm.y * h)

                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (230, 185, 4), cv2.FILLED)
        return lmList


# dummy
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector() #

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img, draw=False)
        print(landmarkList)

        cv2.imshow('Camera', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()