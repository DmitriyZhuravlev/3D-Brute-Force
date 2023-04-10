import cv2
import numpy as np

from library.Math import *
from library.File import *
from library.Plotting import *
from library.lifting_3d  import * 

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    # Range for upper range
    # yellow_lower = np.array([20, 100, 100])
    # yellow_upper = np.array([30, 255, 255])

    lower = np.uint8([ 60,  100, 40])
    upper = np.uint8([ 255,  200, 255])
    #upper = np.uint8([ 50,   50, 100])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = yellow_mask)

origin = cv2.imread('/home/dzhura/Code/yolov7-segmentation/image_4/006745.jpg', cv2.IMREAD_COLOR)


img = select_white_yellow(origin)

#img = cv2.resize(img, dsize=(600, 600))
cv2.imshow("Result Image", img)
cv2.waitKey(0)
img = apply_smoothing(img, 15)
cv2.imshow("Result Image", img)
cv2.waitKey(0)
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Result Image", gray)
cv2.waitKey(0)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 10, 250)
cv2.imshow("Result Image", edges)
cv2.waitKey(0)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250) #np.pi/180, 100, minLineLength=100, maxLineGap=2500)
# Draw lines on the image
ld = (1e09, -1e09)
lu = (1e09, 1e09)

ru = (-1e09, 1e09)
rd = (-1e09, -1e09)
 
rect = []
for line in lines:
    x1, y1, x2, y2 = line[0]

    if xmin > x1: xmin = x1
    if xmin > x2: xmin = x2
    if xmax < x1: xmax = x1
    if xmax < x2: xmax = x2
    
    if ymin > y1: ymin = y1
    if ymin > y2: ymin = y2
    if ymax < y1: ymax = y1
    if ymax < y2: ymax = y2

    rect.append((x1,y1))
    rect.append((x2,y2))

    cv2.line(origin, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.circle(origin, (x1, y1), 5, cv_colors.GREEN.value, thickness=-1)

#rect = [(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)]
#for p in rect()
#cv2.fillConvexPoly(origin, np.int32([rect]), cv_colors.WHITE.value)
#cv2.drawContours(origin, np.int32([rect]), 0, cv_colors.ORANGE.value, 4)

hull = cv2.convexHull(np.int32(np.array(rect)))
cv2.drawContours(origin, [hull], 0, cv_colors.ORANGE.value, 4)
#cv2.fillConvexPoly(origin, hull, cv_colors.WHITE.value)

# Show result
#img = cv2.resize(origin, dsize=(600, 600))
cv2.imshow("Result Image", origin)

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()
