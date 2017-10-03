
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

def pedestrian_detection(imagePath):
    hog = cv2.HOGDescriptor()	#load histogram of oriented gradients descriptor
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())	#apply it to people
    image = cv2.imread(imagePath)	#open image
    image = cv2.resize(image, (400, 300))	#resize
	#rects: array of position
	#weights: probability
	#winstride beyemshy 4,4 in x and y
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	#remove overlapping rectangles
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
	#draw rectangles
        if (yB > 250):
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    print ("Found {0} Pedestrains!".format(len(pick)))
    cv2.imshow("Pedestrains Found", image)

def car_detection(imagePath):
    cascPath = 'cars.xml'	#load the training set file
    carCascade = cv2.CascadeClassifier(cascPath)	#train the classifier
    image = cv2.imread(imagePath)	#read the image
    image = cv2.resize(image, (500, 400))	#resize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 	#turn it to gray
    cars = carCascade.detectMultiScale(gray, 1.1, 1)	#detect the cars in the image
    for (x, y, w, h) in cars:
        if (w > 200):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print("Found {0} Cars!".format(len(cars)))
    cv2.imshow("Cars Found", image)

imagepath = 'test.bmp'
pedestrian_detection(imagepath)
car_detection(imagepath)
cv2.waitKey(0)