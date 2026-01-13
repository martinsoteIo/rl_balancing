import numpy as np
import cv2 as cv
import os
 
script_name = os.path.splitext(os.path.basename(__file__))[0]
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, script_name)
os.makedirs(output_dir, exist_ok=True)

img = cv.imread('test.png')
assert img is not None, "file could not be read, check with os.path.exists()"
cv.imwrite(os.path.join(output_dir, "01_original.png"), img)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imwrite(os.path.join(output_dir, "02_hsv.png"), hsv)

lower_orange = np.array([5, 150, 150])
upper_orange = np.array([25, 255, 255])
mask_color = cv.inRange(hsv, lower_orange, upper_orange)
cv.imwrite(os.path.join(output_dir, "03_mask_color_raw.png"), mask_color)

kernel = np.ones((5,5), np.uint8)
mask_color = cv.morphologyEx(mask_color, cv.MORPH_OPEN,  kernel, iterations=1)
mask_color = cv.morphologyEx(mask_color, cv.MORPH_CLOSE, kernel, iterations=2)
mask_color = cv.medianBlur(mask_color, 5)

circles = cv.HoughCircles(mask_color,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

mask_circle = np.zeros_like(mask_color)
if circles is not None:
    circles = np.uint16(np.around(circles))
    x, y, r = max(circles[0, :], key=lambda c: c[2]) # we select the biggest circle
    cv.circle(mask_circle, (x, y), r, 255, -1)  # -1 = filled
else:
    mask_circle = mask_color
cv.imwrite(os.path.join(output_dir, "04_mask_circle.png"), mask_circle)

gray3 = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
result = cv.bitwise_and(gray3, gray3, mask=mask_circle)
   
if circles is not None:
    # draw the outer circle
    cv.circle(result, (x, y), r,(0,255,0),2)
    # draw the center of the circle
    cv.circle(result, (x, y), 2,(0,0,255),3)
cv.imwrite(os.path.join(output_dir, "05_result.png"), result)
 
cv.imshow('detected ball circle',result)
cv.waitKey(0)
cv.destroyAllWindows()