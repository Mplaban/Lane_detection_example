import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def make_coord(image,line_parameter):
	print("Enter")
	slope,intercept=line_parameter
	y1 = image.shape[0]
	y2 = int(y1*1.8/5)
	x1 = int((y1-intercept)/slope)
	x2 = int((y2-intercept)/slope)
	return np.array([x1,y1,x2,y2])

def canny(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(blur,30,89)
	return canny

def display_lines(image,lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2=line.reshape(4)
			parameters =  np.polyfit((x1,x2),(y1,y2),1)
			slope = parameters[0]
			if slope < 0:
				cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
			else:
				cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)		    
	return line_image

def average_slope_intercept(image,lines):
	right_fit=[]
	left_fit = []
#	last_left_line=[-335  ,352 , 260  ,126]
#	last_right_line=[-335 , 352  ,224 , 126]
	if lines is not None:
		for line in lines:
			x1 , y1 ,x2 ,y2 = line.reshape(4)
			parameters = np.polyfit((x1,x2),(y1,y2),1)
			slope = parameters[0]
			intercept = parameters[1]
			if slope < 0:
				left_fit.append((slope,intercept))
			else:
				right_fit.append((slope,intercept))

		left_fit_average = np.average(left_fit,axis=0)
		print(left_fit_average)
		print(np.isnan(left_fit_average))
		if np.isnan(left_fit_average)[1] != True :
			print("YES")
			left_fit_average=[ -0.40349224 ,216.59467849]
		
		right_fit_average = np.average(right_fit,axis=0)
		if (np.isnan(left_fit_average)) is not [True , True] :
			print("Left")
			left_line=make_coord(image,left_fit_average)
			last_left_line=left_line
			print(left_fit_average)

		if (np.isnan(right_fit_average)) is not [True,True] :
			print("Right")
			right_line=make_coord(image,right_fit_average)
			last_right_line=right_line
#			print(left_fit_average)
		
		left_line=last_left_line
		right_line=last_right_line
	return np.array([left_line,right_line])


def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
	[(4,200),(620,height),(320,13)]
	])
	#polygons = np.array([
	#[(13,680),(1500,610),(950,280)]
	#])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask,polygons,255)
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image


#image = cv2.imread('etse.jpg')
#lane_image = np.copy(image)
#canny_image = canny(lane_image)
#cropped = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped,2,np.pi/180,80,np.array([]),minLineLength=30,maxLineGap=8)
#average_lines = average_slope_intercept(lane_image,lines)
#line_image = display_lines(lane_image,average_lines)
#combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)


#cv2.imshow("result",combo_image)
#cv2.waitKey(0)

cap = cv2.VideoCapture("test3.1.mp4")
while(cap.isOpened()):
	ret,frame = cap.read()
	canny_image = canny(frame)
	cropped = region_of_interest(canny_image)
	lines = cv2.HoughLinesP(cropped,2,np.pi/180,100,np.array([]),minLineLength=60,maxLineGap=3)
	average_lines = average_slope_intercept(frame,lines)
	line_image = display_lines(frame,average_lines)
	combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)
	cv2.imshow("Result",combo_image)
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()





















