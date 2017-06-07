import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import RPi.GPIO as GPIO

GPIO.cleanup()
GPIO.setmode(GPIO.BCM)

LEFT_BIT_0 = 2
LEFT_BIT_1 = 3
LEFT_BIT_2 = 4

RIGHT_BIT_0 = 17
RIGHT_BIT_1 = 27
RIGHT_BIT_2 = 22

#left wheel
GPIO.setup(LEFT_BIT_0, GPIO.OUT)
GPIO.setup(LEFT_BIT_1, GPIO.OUT)
GPIO.setup(LEFT_BIT_2, GPIO.OUT)

#right wheel 
GPIO.setup(RIGHT_BIT_0, GPIO.OUT)
GPIO.setup(RIGHT_BIT_1, GPIO.OUT)
GPIO.setup(RIGHT_BIT_2, GPIO.OUT)


cap = cv2.VideoCapture(0)
CIRCLE_CORRECTION_FACTOR = 1.2675 #needed for comparison of area of circles

#cap.set(3,1280)
#cap.set(4,720)

def setWheels(direction):
	
	if(area <1  or roundness < 0.2):
	#	print "no ball, stop"
		GPIO.output(RIGHT_BIT_0, GPIO.LOW)
		GPIO.output(RIGHT_BIT_1, GPIO.LOW)
		GPIO.output(RIGHT_BIT_2, GPIO.LOW)
		
		GPIO.output(LEFT_BIT_0, GPIO.LOW)
		GPIO.output(LEFT_BIT_1, GPIO.LOW)
		GPIO.output(LEFT_BIT_2, GPIO.LOW)
	elif(direction<20 and direction>-20): #go straight
		GPIO.output(RIGHT_BIT_0, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_1, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_2, GPIO.LOW)
		
		GPIO.output(LEFT_BIT_0, GPIO.HIGH)
		GPIO.output(LEFT_BIT_1, GPIO.HIGH)
		GPIO.output(LEFT_BIT_2, GPIO.LOW)
	#	print "Going straight"
		 
	elif(direction<-20 and direction>-120): #go slow left
		GPIO.output(RIGHT_BIT_0, GPIO.LOW)
		GPIO.output(RIGHT_BIT_1, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_2, GPIO.LOW)
		
		GPIO.output(LEFT_BIT_0, GPIO.HIGH)
		GPIO.output(LEFT_BIT_1, GPIO.HIGH)
		GPIO.output(LEFT_BIT_2, GPIO.LOW)
	#	print "slow left"
		
		
	elif(direction<-120 and direction>-220): #go medium left
		GPIO.output(RIGHT_BIT_0, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_1, GPIO.LOW)
		GPIO.output(RIGHT_BIT_2, GPIO.LOW)
		
		GPIO.output(LEFT_BIT_0, GPIO.HIGH)
		GPIO.output(LEFT_BIT_1, GPIO.HIGH)
		GPIO.output(LEFT_BIT_2, GPIO.LOW)
	#	print "medium left"
		
	elif(direction<-220 and direction>-320): #go fast left
		GPIO.output(RIGHT_BIT_0, GPIO.LOW)
		GPIO.output(RIGHT_BIT_1, GPIO.LOW)
		GPIO.output(RIGHT_BIT_2, GPIO.LOW)
		
		GPIO.output(LEFT_BIT_0, GPIO.HIGH)
		GPIO.output(LEFT_BIT_1, GPIO.HIGH)
		GPIO.output(LEFT_BIT_2, GPIO.LOW)
	#	print "fast left"
		
	elif(direction<120 and direction>20): #go slow right
		GPIO.output(RIGHT_BIT_0, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_1, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_2, GPIO.LOW)
		
		GPIO.output(LEFT_BIT_0, GPIO.LOW)
		GPIO.output(LEFT_BIT_1, GPIO.HIGH)
		GPIO.output(LEFT_BIT_2, GPIO.LOW)
	#	print "slow right"
		
	elif(direction<220 and direction>120): #go medium right
		GPIO.output(RIGHT_BIT_0, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_1, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_2, GPIO.LOW)
		
		GPIO.output(LEFT_BIT_0, GPIO.HIGH)
		GPIO.output(LEFT_BIT_1, GPIO.LOW)
		GPIO.output(LEFT_BIT_2, GPIO.LOW)
	#	print "medium right"
		
	elif(direction<320 and direction>220): #go fast right
		GPIO.output(RIGHT_BIT_0, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_1, GPIO.HIGH)
		GPIO.output(RIGHT_BIT_2, GPIO.LOW)
		
		GPIO.output(LEFT_BIT_0, GPIO.LOW)
		GPIO.output(LEFT_BIT_1, GPIO.LOW)
		GPIO.output(LEFT_BIT_2, GPIO.LOW)
	#	print "fast right"
		
		
def fitnessFunction(size, roundness):
  if size<10:
    return -100000000.0
  else:
    return 2*roundness + size/80000.

def roundnessFunction(contour):
  area = cv2.contourArea(contour)
  
  if (area > 10):
    (x,y),radius  = cv2.minEnclosingCircle(contour)
    idealArea = math.pi*radius**2.
    #print "area: " + str(area) + " ideal : " +str(idealArea) 
    return area/idealArea
  else:
    return 0

###old roundness function
      #areas[i] = (cv2.contourArea(contours[i]))
      #circumferences[i] = (cv2.arcLength(contours[i],True))
      #calculatedAreas[i] = (math.pow(circumferences[i],2)/(4*math.pi)/CIRCLE_CORRECTION_FACTOR)
      #roundnesses[i] = (areas[i]/calculatedAreas[i])

centerH = 120 
deltaH = 10

while(True):
  ret, frame = cap.read()
  #frameBlurred = cv2.medianBlur(frame,5)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
  # define range of blue color in HSV
#  lower_blue = np.array([centerH-deltaH,100,100])
 # upper_blue = np.array([centerH+deltaH,255,255])

  #lower_red2 = np.array([160,30,30])
  #upper_red2 = np.array([180,150,255])
  
  #lower_red = np.array([0,30,30])
  #upper_red = np.array([10,255,255])
  
  lower_orange = np.array([160, 40, 150])
  upper_orange = np.array([180,255,255])
  
  #print hsv[640/2][480/2]
  #print "Center color HSV" + str(hsv[640/2][480/2])

  # Threshold the HSV image to get only blue colorsx
  mask = cv2.inRange(hsv, lower_orange, upper_orange)
  #mask = cv2.inRange(hsv, lower_red, upper_red)
  #mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_red2, upper_red2))
  
  moment = cv2.moments(mask, True);
  area = moment['m00']
  
  #im2, contours, hierarchy = cv2.findContours(mask, 1, 2)
  maskCoppy = np.copy(mask)
  contours, hierarchy = cv2.findContours(maskCoppy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  mostRound = 0
  #biggest
  
  #in case there are no contours
  if(len(contours) > 0):
    areas = np.zeros(len(contours), dtype=np.float64)
    roundnesses = np.zeros(len(contours), dtype=np.float64)
    for i in range(0,len(contours)):
      roundnesses[i] = roundnessFunction(contours[i])
      areas[i] = (cv2.contourArea(contours[i]))
      if(fitnessFunction(areas[i], roundnesses[i]) > fitnessFunction(areas[mostRound], roundnesses[mostRound])):
        mostRound = i
    
    perimiterCont = contours[mostRound]
    circumference = (cv2.arcLength(perimiterCont,True))
    area = (cv2.contourArea(perimiterCont))
    roundness = roundnesses[mostRound]
  else:
    perimiterCont = np.array([[0,0],[0,0]])
    circumference = 0
    area = 0
    calculatedArea = 0
    roundness = 0
  
  
  
  
  #print "number of contures: " + str(len(contours)) + " Area: " + str(area) + " Roundness: " + str(roundness)
  
  
  #get the center of the idea circle around the circle
  (x,y),radius = cv2.minEnclosingCircle(perimiterCont)
  cx = int(x)
  cy = int(y)
  
#  candidateH = hsv[cy][cx][0]
  
#  if(candidateH<centerH+deltaH and candidateH>centerH-deltaH and roundness>0.8):
#	  centerH = math.floor(0.8*centerH+0.2*candidateH)
	  
  
  #print "x= " + str(cx) + " y= " + str(cy)
  
  cv2.circle(frame, (cx,cy), 5, (0,255,0))
  cv2.circle(frame, (640/2,480/2), 2, (255,0,0))
  
  direction = cx-320
  
  setWheels(direction)
  
  #mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_red, upper_red))

  # Bitwise-AND mask and original image
  #res = cv2.bitwise_and(frame,frame, mask= mask)
  
  #edge = cv2.Laplacian(mask,cv2.CV_64F)

  cv2.drawContours(frame, [perimiterCont], 0, (0,255,0), 3)
  #cv2.imshow('frame',frame)
  #cv2.imshow('mask',mask)
  #cv2.imshow('res',res)
  #cv2.imshow('edge',edge)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
GPIO.cleanup()
