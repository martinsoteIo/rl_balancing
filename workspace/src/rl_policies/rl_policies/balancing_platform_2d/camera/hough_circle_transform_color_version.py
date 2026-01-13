import numpy as np
import cv2 as cv
import os
 
script_name = os.path.splitext(os.path.basename(__file__))[0]
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, script_name)
os.makedirs(output_dir, exist_ok=True)

def save(name, img):
    cv.imwrite(os.path.join(output_dir, name), img)
 
img = cv.imread('test.png')
assert img is not None, "file could not be read, check with os.path.exists()"
save("01_original.png", img)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
save("02_hsv.png", hsv)

lower_orange = np.array([5, 150, 150])
upper_orange = np.array([25, 255, 255])
mask = cv.inRange(hsv, lower_orange, upper_orange)
save("03_mask_raw.png", mask)

mask = cv.medianBlur(mask,5)
gray = mask
 
circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    save("04_circles.png", img)
    # draw the center of the circle
    cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
save("05_result.png", img)
 
cv.imshow('detected ball circle',img)
cv.waitKey(0)
cv.destroyAllWindows()