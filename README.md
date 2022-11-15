# TuxasHandtracking
## Instructions:

Requirements:
```
pip install opencv
```

Clone from git
> Terminal
```
git clone https://github.com/Tuxa4Life/TuxasHandtracking.git
```

Then import to a file
> Code
``` 
import TuxasHandtracking as th
```

# Function parameters
```
# class constructor
__init__(self, mode=False, maxHands=2, complexity=1, detectionConfidence=0.5, trackConfidence=0.5)

# finding hands
findHands(self, img, draw=True)

# finding posotions of landmarks
findPosition(self, img, handId=0, draw=True)
```


## Example
```
# importing modules
import TuxasHandtracking as th
import cv2
import time

cap = cv2.VideoCapture(0) # getting video data
detector = th.HandDetector() # importing HandDetector class

while True:
    success, img = cap.read() # getting video

    img = detector.findHands(img) # scanning hands and drawing landmarks
    landmarkList = detector.findPosition(img, draw=False) # getting landmark positions

    print(landmarkList)

    cv2.imshow('Camera', img)
    cv2.waitKey(1)
```
