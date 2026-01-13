import numpy as np
import cv2 as cv

class BallDetector:
    def __init__(self,
                 lower_hsv=(5, 150, 150),
                 upper_hsv=(30, 255, 255),
                 kernel_size=5,
                 dp=1, minDist=20, param1=50, param2=30,
                 minRadius=0, maxRadius=0):
        # HSV thresholds for orange ball
        self.lower = np.array(lower_hsv, dtype=np.uint8)
        self.upper = np.array(upper_hsv, dtype=np.uint8)
        # Morphological kernel
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Hough parameters
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.minRadius = minRadius
        self.maxRadius = maxRadius

    def detect(self, img_bgr, gray_background=True):
        """
        Detect a colored ball using HSV mask + contours (robust).
        Returns:
            result (np.ndarray): output image with ball and black/gray background
            circle (np.ndarray or None): [x, y, r] of the ball
            mask_color (np.ndarray): binary mask from HSV threshold
            mask_circle (np.ndarray): circular mask of the detected ball
        """
        # Convert to HSV and apply color threshold
        hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
        mask_color = cv.inRange(hsv, self.lower, self.upper)
        mask_color = cv.morphologyEx(mask_color, cv.MORPH_OPEN, self.kernel, iterations=1)
        mask_color = cv.morphologyEx(mask_color, cv.MORPH_CLOSE, self.kernel, iterations=2)
        mask_color = cv.medianBlur(mask_color, 5)

        circle = None
        mask_circle = np.zeros_like(mask_color)

        # Detect largest contour and fit enclosing circle
        cnts, _ = cv.findContours(mask_color, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv.contourArea)
            (fx, fy), fr = cv.minEnclosingCircle(c)
            x, y, r = int(fx), int(fy), int(fr)
            if r >= max(5, self.minRadius):  # filter small noise
                circle = np.array([x, y, r], dtype=np.int32)
                cv.circle(mask_circle, (x, y), r, 255, -1)
        else:
            # Fallback: use plain color mask if no circle is found
            mask_circle = mask_color

        # Create output with black/gray background
        if gray_background:
            gray3 = cv.cvtColor(cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
            result = cv.bitwise_and(gray3, gray3, mask=mask_circle)
        else:
            result = cv.bitwise_and(img_bgr, img_bgr, mask=mask_circle)

        # Draw circle outline and center
        if circle is not None:
            x, y, r = map(int, circle)
            cv.circle(result, (x, y), r, (0, 255, 0), 2)
            cv.circle(result, (x, y), 2, (0, 0, 255), 3)

        return result, circle, mask_color, mask_circle


