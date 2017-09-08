# import the necessary packages
from collections import deque
import numpy as np
import imutils
import cv2
import threading


width = 25
left  = 114
erode = 10

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
lower = (227,255, 255)
upper = (21, 50, 50)


class Reward(threading.Thread):
    def __init__(self, bufferSize=5, rewardFactor=10, *args, **kwargs):
        self.bufferSize = bufferSize
        self.rewardFactor = rewardFactor
        super().__init__(*args, **kwargs)

    def getReward(self):
        return self.currentReward

    def run(self):
        pts = deque(maxlen=self.bufferSize)

        camera = cv2.VideoCapture(CAMERA)

        def compFilter(prev, next, percent):
            return prev * percent + (1 - percent) * next
# keep looping
        x0, y0 = 0, 0
        oldcenter = 0,0
        while True:
            last = lower, upper
            upper = ((left + width) % 180, 255, 255)
            lower = (left % 180, 50, 50)
            (grabbed, frame) = camera.read()

            # resize the frame, blur it, and convert it to the HSV
            # color space
            frame = imutils.resize(frame, width=1280)
            blur = cv2.GaussianBlur(frame, (21, 21), 0)
            colormode = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV) # it's actually bgr but whatever

            # construct a mask for the color "green", then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask
            mask = cv2.inRange(colormode, lower, upper)
            mask = cv2.erode(mask, None, iterations=erode)
            mask = cv2.dilate(mask, None, iterations=2)
            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None

            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    x0 = compFilter(x0, x, 0.68)
                    y0 = compFilter(y0, y, 0.68)
                    x, y = x0, y0
                    M = cv2.moments(c)
                    center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                    oldcenter = compFilter(oldcenter[0], center[0], 0.68), compFilter(oldcenter[1],center[1], 0.68)
                    center = int(oldcenter[0]), int(oldcenter[1])

            # update the points queue
            pts.appendleft(center)
            currentpos = center
            oldpos = pts[1]
            self.currentReward = np.tanh(self.rewardFactor * (mag(currentpos - self.origin) - mag(oldpos - self.origin)))

        camera.release()
        cv2.destroyAllWindows()
