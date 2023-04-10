import argparse
import os
import platform
import sys
from pathlib import Path
from library.Math import *
from library.File import *
from library.Plotting import *
#from library.lifting_3d  import * 
#from library.augmentation import *
from bisect import insort

import torch
import torch.backends.cudnn as cudnn

#..... Tracker modules......
import skimage
from sort_count import *
import numpy as np
#...........................

debug = False
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,scale_segments, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode



class Solution:
    def __init__(self, loc, orient, over_error, error, iou = 0):
        self.loc = loc
        self.orient = orient
        self.over_error = over_error
        self.error = error
        self.iou = iou

    def __lt__(self, other):
        return self.over_error < other.over_error

def binaryMaskIOU_old(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 != (0, 0, 0))
    mask2_area = np.count_nonzero(mask2 != (0, 0, 0))
    intersection = np.count_nonzero(np.logical_and(mask1 != (0, 0, 0),  mask2 != (0, 0, 0)))
    # print(mask1_area)
    # print(mask2_area)
    # print(intersection)
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


# def det_lines(img, img_out, yolo_mask, box_2d, theta, optic_center):
    # e = 1
    # xmin = int(box_2d[0][0]) + e
    # ymin = int(box_2d[0][1]) + e
    # xmax = int(box_2d[1][0]) - e
    # ymax = int(box_2d[1][1]) - e
    
    # img = img[ymin:ymax, xmin:xmax]
    # img_drow = img_out[ymin:ymax, xmin:xmax]
    # mask = yolo_mask[ymin:ymax, xmin:xmax]
    
    # # Convert the image to gray-scale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow("gray", gray)
    # # cv2.waitKey(0)
    
    
    # # Otsu's thresholding
    # ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # # cv2.imshow("Otsu", th2)
    # # cv2.waitKey(0)
    # # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # # cv2.imshow("Otsu2", th3)
    # # cv2.waitKey(0)
    # # Find the edges in the image using canny detector
    # edges = cv2.Canny(th3, 10, 250)
    # # cv2.imshow("edges", edges)
    # # cv2.waitKey(0)
    
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=10)
    # lx1  = None
    # lx2  = None
    # ly1  = None
    # ly2 = None
    # maxd = -1e09
    # if lines is None: return
    # for line in lines:
        # x1, y1, x2, y2 = line[0]
        # d = math.dist((x1, y1), (x2, y2))
        # if maxd < d:
            # maxd = d
            # lx1 = x1
            # lx2 = x2
            # ly1 = y1
            # ly2 = y2
        # cv2.line(img, (x1, y1), (x2, y2), cv_colors.PURPLE.value, 3)

    # cv2.line(img, (lx1, ly1), (lx2, ly2), cv_colors.ORANGE.value, 6)
    # cv2.line(img_drow, (lx1, ly1), (lx2, ly2), cv_colors.ORANGE.value, 6)
    # cv2.line(img_out, optic_center, (xmin + lx1, ymin + ly1), cv_colors.MINT.value, 4)
    # # cv2.imshow("lines", img)
    # # cv2.waitKey(0)
    
    # alpha = np.pi + get_angle(optic_center, (xmin + lx1, ymin + ly1), (xmin + lx2, ymin + ly2))
    
    # print("local angle: ", np.degrees(alpha))
    # print("local angle: ", np.degrees(np.pi/2 - alpha))
    # print("theta angle: ", np.degrees(theta))
    # print("global angle: ", (np.degrees(alpha) + np.degrees(theta)) % 180)
    # print("global angle: ", (np.degrees(alpha) + 90 + np.degrees(theta)) % 180)
    
    # cv2.imshow("lines", img)
    # cv2.imshow("res", img_out)
    # cv2.waitKey(0)
    
    # return [alpha] #, -alpha] #, -alpha + np.pi/2]


# def get_line(img, img_out, yolo_mask, box_2d, theta, optic_center):
    # e = 1
    # xmin = int(box_2d[0][0]) + e
    # ymin = int(box_2d[0][1]) + e
    # xmax = int(box_2d[1][0]) - e
    # ymax = int(box_2d[1][1]) - e
    
    # img = img[ymin:ymax, xmin:xmax]
    # img_drow = img_out[ymin:ymax, xmin:xmax]
    # mask = yolo_mask[ymin:ymax, xmin:xmax]
    
    # # Convert the image to gray-scale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow("gray", gray)
    # # cv2.waitKey(0)
    
    
    # # Otsu's thresholding
    # ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # # cv2.imshow("Otsu", th2)
    # # cv2.waitKey(0)
    # # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # # cv2.imshow("Otsu2", th3)
    # # cv2.waitKey(0)
    # # Find the edges in the image using canny detector
    # edges = cv2.Canny(th3, 10, 250)
    # # cv2.imshow("edges", edges)
    # # cv2.waitKey(0)
    
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=10)
    # lx1  = None
    # lx2  = None
    # ly1  = None
    # ly2 = None
    # maxd = -1e09
    # if lines is None: return
    # for line in lines:
        # x1, y1, x2, y2 = line[0]
        # d = math.dist((x1, y1), (x2, y2))
        # if maxd < d:
            # maxd = d
            # lx1 = x1
            # lx2 = x2
            # ly1 = y1
            # ly2 = y2
        # cv2.line(img_drow, (x1, y1), (x2, y2), cv_colors.PURPLE.value, 3)

    # cv2.line(img, (lx1, ly1), (lx2, ly2), cv_colors.ORANGE.value, 6)
    # cv2.line(img_drow, (lx1, ly1), (lx2, ly2), cv_colors.ORANGE.value, 6)
    # #cv2.line(img_out, optic_center, (xmin + lx1, ymin + ly1), cv_colors.MINT.value, 4)
    # # cv2.imshow("lines", img)
    # # cv2.waitKey(0)
    
    # alpha = np.pi + get_angle(optic_center, (xmin + lx1, ymin + ly1), (xmin + lx2, ymin + ly2))
    
    # print("local angle: ", np.degrees(alpha))
    # print("local angle: ", np.degrees(np.pi/2 - alpha))
    # print("theta angle: ", np.degrees(theta))
    # print("global angle: ", (np.degrees(theta) + np.degrees(theta)) % 180)
    # print("global angle: ", (np.degrees(theta) + 90 + np.degrees(theta)) % 180)
    
    # # cv2.imshow("lines", img)
    # # cv2.imshow("res", img_out)
    # # cv2.waitKey(0)
    
    # return [(xmin + lx1, ymin + ly1), (xmin + lx2, ymin + ly2)]


def binaryMaskIOU(mask1, mask2, box_2d, eps=1e-7):
    xmin = int(box_2d[0][0])
    ymin = int(box_2d[0][1])
    xmax = int(box_2d[1][0])
    ymax = int(box_2d[1][1])
    
    mask1 = mask1[ymin:ymax, xmin:xmax]
    mask2 = mask2[ymin:ymax, xmin:xmax]
    
    # numpy_vertical = np.concatenate((mask1, mask2), axis=0)
    # cv2.imshow('Mask', numpy_vertical)
    # cv2.waitKey(0)

    mask1_area = np.count_nonzero(mask1 != (0, 0, 0))
    mask2_area = np.count_nonzero(mask2 != (0, 0, 0))
    intersection = np.count_nonzero(np.logical_and(mask1 != (0, 0, 0),  mask2 != (0, 0, 0)))
    # print(mask1_area)
    # print(mask2_area)
    # print(intersection)
    iou = intersection /(mask1_area+mask2_area-intersection + eps)
    #print("IOU :", iou)
    return iou


def calc_theta_ray(img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle
# # class names
# names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         # 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         # 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         # 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         # 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         # 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         # 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         # 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         # 'hair drier', 'toothbrush' ]

# vehicles = ['car', 'truck', 'van', 'tram','pedestrian','cyclist']
# dims_avg = {'cyclist': np.array([ 1.73532436,  0.58028152,  1.77413709]), 'van': np.array([ 2.18928571,  1.90979592,  5.07087755]),
            # 'tram': np.array([  3.56092896,   2.39601093,  18.34125683]), 'car': np.array([ 1.52159147,  1.64443089,  3.85813679]),
            # 'pedestrian': np.array([ 1.75554637,  0.66860882,  0.87623049]), 'truck': np.array([  3.07392252,   2.63079903,  11.2190799 ])}

def get_dim(class_):
        if class_ == 0: # pedestrian
            return [ 1.75554637,  0.66860882,  0.87623049]
        if class_ == 2: # car
            return [ 1.52159147,  1.64443089,  3.85813679]
        if class_ == 7: #truck
            return [  3.07392252,   2.63079903,  11.2190799 ] #[3.09, 2.6 , 16.1544]
        if class_ == 5: # bus
            return [  3.56092896,   2.39601093,  18.34125683]

        return None

def get_dim_custom(class_):
        if class_ == 0:
            return [1.77611111, 0.65944444, 0.83666667]
        if class_ == 2:
            return [1.511875,   1.62678571, 3.79705357]
        if class_ == 7:
            #return [ 3.09,        2.38714286, 10.70571429]
            return [2.9, 2.6 , 16.1544] #[3.09, 2.6 , 16.1544]

        return None

def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)

def plot_regressed_3d_bbox_mod(img, cam_to_img, cls, box_2d, yolo_mask):#, location):
    xmin = int(box_2d[0][0])
    ymin = int(box_2d[0][1])
    xmax = int(box_2d[1][0])
    ymax = int(box_2d[1][1])
    
    # padding = np.zeros_like(yolo_mask)
    # padding[ymin:ymax, xmin:xmax] = yolo_mask[ymin:ymax, xmin:xmax]
    yolo_mask = yolo_mask[ymin:ymax, xmin:xmax]
    
    # padding = yolo_mask
    # img = padding
    # convert the input image to grayscale
    gray = cv2.cvtColor(yolo_mask, cv2.COLOR_BGR2GRAY)
    

    # # apply thresholding to convert grayscale to binary image
    # ret,thresh = cv2.threshold(gray,150,255,0)
    
    # find the contours
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).

    # Find the contour with the maximum area.
    cnt = max(cnts, key=cv2.contourArea)
    
    hull = cv2.convexHull(cnt)

    points = []
    index = hull[:,:,1].argmax()
    points.append(tuple(hull[index][0]))
    #np.delete(hull, np.argwhere(hull[index][0]) == 6) [index])


    index = hull[:,:,1].argmax()
    points.append(tuple(hull[index][0]))
    
    print(points)

    bgr_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.circle(bgr_img, points[0], 1, cv_colors.RED.value, -1)
    cv2.circle(bgr_img, points[1], 1, cv_colors.YELLOW.value, -1)
    cv2.imshow('bgr_img', bgr_img)
    cv2.waitKey()    

    # find the extreme points
    # leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    # rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    # topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    # bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    # points = [leftmost, rightmost, topmost, bottommost]
    
    # # draw the points on th image
    # for point in points:
       # cv2.circle(img, (point[0] + xmin, point[1] + ymin), 4, cv_colors.RED.value, -1)

    #numpy_vertical = np.concatenate((yolo_mask, gray), axis=0)
    # cv2.imshow('Mask', yolo_mask)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    return None, None #variants[0].loc


def plot_regressed_3d_bbox_test(img_in, img, cam_to_img, cls, box_2d, yolo_mask, optic_center):
    if debug:
        print(box_2d)
        print("cam_to_img", cam_to_img)
        print("dimensions", dimensions)



    best_loc = None
    best_error = [1e09]
    best_over_error = [1e09]
    best_X = None
    best_orient = None
    best_iot = 0
    #print("dim :", dimensions)
    variants = []
    

    
    dim = get_dim_custom(cls)
    #v, t = get_model(cls)
    
    theta = calc_theta_ray(img, box_2d, cam_to_img)
    angles = det_lines(img_in, img, yolo_mask, box_2d, theta, optic_center)
    if angles is None: return None, None, None

    plot_2d_box(img, box_2d, 4)
    # the math! returns X, the corners used for constraint
    #for orient_deg in range(0, 179, 1):
    #for alpha in [rot_angle, rot_angle + np.pi/2]:
    for alpha in angles:

        #orient = np.radians(orient_deg)
        orient = alpha + theta
        #print("alpha :", alpha_deg)
        if True: #try:
            location, X, over_error, error = calc_location_test(dim, cam_to_img, box_2d, theta, orient - theta)
            if location is not None:
                insort(variants, Solution(location, orient, over_error, error))
                plot_3d_box(img, cam_to_img, orient, dim, location, thickness = 2)

    variants = variants[:4]
    #variants.sort(reverse = False, key = lambda x: x.error)
    #print("variants :", variants)
    
    # if len(variants) > 0:
        # plot_3d_box(img, cam_to_img, variants[0].orient, dim, variants[0].loc, 5)
    
# if debug:
    for var in variants[0:4]:
        #plot_3d_box(img, cam_to_img, v.orient, dim, v.loc)

        c_mask = np.zeros(img.shape, np.uint8)
        get_3d_mask(c_mask, cam_to_img, var.orient, dim, var.loc) #, cls, v, t)
        var.iou =  binaryMaskIOU(c_mask, yolo_mask, box_2d)

    if len(variants) > 0:
        color = list(np.random.random(size=3) * 256)
       # variants.sort(reverse = True, key = lambda x: x.iou)
        lower_face = plot_3d_box(img, cam_to_img, variants[0].orient, dim, variants[0].loc, thickness = 3,  color = cv_colors.RED.value)
        print("IOU : ", variants[0].iou)
        print("best local orient: ", variants[0].orient - theta)
        print("best global orient: ", variants[0].orient)
        
        cv2.waitKey(0)

        return variants[0].loc, lower_face, color
        

    return None, None, None #variants[0].loc


def plot_regressed_3d_bbox_fast(img_in, img, cam_to_img, cls, box_2d, yolo_mask, optic_center, mat, inv_mat):#, location):
    #if angles is None: return None, None, None
    if debug:
        print(box_2d)
        print("cam_to_img", cam_to_img)
        print("dimensions", dimensions)

    #plot_2d_box(img, box_2d, thickness = 3)
    plot_2d_box(img, box_2d, 4)
        
    #rot_angle = plot_regressed_3d_bbox_mod(img, cam_to_img, cls, box_2d, yolo_mask)


    best_loc = None
    best_error = [1e09]
    best_over_error = [1e09]
    best_X = None
    best_orient = None
    best_iot = 0
    #print("dim :", dimensions)
    variants = []
    
    dim = get_dim_custom(cls)
    #v, t = get_model(cls)
    
    theta = calc_theta_ray(img, box_2d, cam_to_img)
    
    line = get_line(img_in, img, yolo_mask, box_2d, theta, optic_center)
    if line is None: return None, None, None
    
    # normal = (line[1][0], line[1][1] - 1)
    # line.append(normal)
    
    # angle  = get_angle_list(line)
    # print("angle :", np.degrees(angle))

    line_bev = to_warp(line, mat)
    line_bev.append((line_bev[1][0], line_bev[1][1] - 1))
    angle_bev  = get_angle_list(line_bev)
    print("angle bev :", np.degrees(angle_bev))

    print("line :", line)
    print("line_bev :", line_bev)
    angles = [angle_bev, angle_bev + np.pi/2] #, angle_bev + np.radians(10), angle_bev + np.radians(-10), angle_bev + np.pi/2 + np.radians(10), angle_bev + np.pi/2 + np.radians(-10) ]
    #return None, None, None
    # the math! returns X, the corners used for constraint
    for orient in angles:
    #for alpha in [rot_angle, rot_angle + np.pi/2]:

        #orient = np.radians(orient_deg)
        #print("alpha :", alpha_deg)
        if True: #try:
            location, X, over_error, error = calc_location_test(dim, cam_to_img, box_2d, theta, orient - theta)
            if location is not None:
                insort(variants, Solution(location, orient, over_error, error))
                #plot_3d_box(img, cam_to_img, orient, dim, location, thickness = 2)

    variants = variants[:10]
    #variants.sort(reverse = False, key = lambda x: x.error)
    #print("variants :", variants)
    
    # if len(variants) > 0:
        # plot_3d_box(img, cam_to_img, variants[0].orient, dim, variants[0].loc, 5)
    
# if debug:
    for var in variants[0:10]:
        #plot_3d_box(img, cam_to_img, v.orient, dim, v.loc)

        c_mask = np.zeros(img.shape, np.uint8)
        get_3d_mask(c_mask, cam_to_img, var.orient, dim, var.loc) #, cls, v, t)
        var.iou =  binaryMaskIOU(c_mask, yolo_mask, box_2d)

    if len(variants) > 0:
        color = list(np.random.random(size=3) * 256)
        variants.sort(reverse = True, key = lambda x: x.iou)
        lower_face = plot_3d_box(img, cam_to_img, variants[0].orient, dim, variants[0].loc, thickness = 3, color = color) # , color = cv_colors.RED.value)
        print("IOU : ", variants[0].iou)

        return variants[0].loc, lower_face, color
        

    return None, None, None #variants[0].loc
def regress_location(img, cam_to_img, cls, box_2d, yolo_mask, est_center):
# the math! returns X, the corners used for constraint
    best_iou = 0
    best_loc = None
    dim = get_dim(cls)

    for orient_deg in range(0, 179, 1):
        orient = np.radians(orient_deg)
        for x in range (-50, 50, 2):
            for y in range(-50, 50, 2):
                for z in range(-50, 50, 2):
                    loc = [est_center[0] + x, est_center[1] + y, est_center[2] + z]
                    c_mask = np.zeros(img.shape, np.uint8)
                    get_3d_mask(c_mask, cam_to_img, orient, dim, loc) #, cls, v, t)
                    iou =  binaryMaskIOU(c_mask, yolo_mask, box_2d)
                    if best_iou < iou:
                        best_iou = iou
                        best_loc = loc
                        
    return best_loc


def plot_regressed_3d_bbox(img_in, img, cam_to_img, cls, box_2d, yolo_mask):
    if debug:
        print(box_2d)
        print("cam_to_img", cam_to_img)
        print("dimensions", dimensions)



    best_loc = None
    best_error = [1e09]
    best_over_error = [1e09]
    best_X = None
    best_orient = None
    best_iot = 0
    #print("dim :", dimensions)
    variants = []
    

    
    dim = get_dim_custom(cls)
    #v, t = get_model(cls)
    
    theta = calc_theta_ray(img, box_2d, cam_to_img)
    plot_2d_box(img, box_2d, 4)
    # the math! returns X, the corners used for constraint
    for orient_deg in range(0, 179, 1):
    #for alpha in [rot_angle, rot_angle + np.pi/2]:
    #for alpha in angles:

        orient = np.radians(orient_deg)
        #orient = alpha + theta
        #print("alpha :", alpha_deg)
        if True: #try:
            location, X, over_error, error = calc_location_test(dim, cam_to_img, box_2d, theta, orient - theta)
            if location is not None:
                insort(variants, Solution(location, orient, over_error, error))
                #plot_3d_box(img, cam_to_img, orient, dim, location, thickness = 2)

    variants = variants[:10]
    #variants.sort(reverse = False, key = lambda x: x.error)
    #print("variants :", variants)
    
    # if len(variants) > 0:
        # plot_3d_box(img, cam_to_img, variants[0].orient, dim, variants[0].loc, 5)
    
# if debug:
    for var in variants[0:10]:
        #plot_3d_box(img, cam_to_img, v.orient, dim, v.loc)

        c_mask = np.zeros(img.shape, np.uint8)
        get_3d_mask(c_mask, cam_to_img, var.orient, dim, var.loc) #, cls, v, t)
        var.iou =  binaryMaskIOU(c_mask, yolo_mask, box_2d)

    if len(variants) > 0:
        color = list(np.random.random(size=3) * 256)
       # variants.sort(reverse = True, key = lambda x: x.iou)
        lower_face = plot_3d_box(img, cam_to_img, variants[0].orient, dim, variants[0].loc, thickness = 3,  color = color)
        print("IOU : ", variants[0].iou)
        print("best local orient: ", variants[0].orient - theta)
        print("best global orient: ", variants[0].orient)
        
        cv2.waitKey(0)

        return variants[0].loc, lower_face, variants[0].orient
        

    return None, None, None #variants[0].loc


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        trk = False,
): 
    
    
    # global TARGET_H
    # global TARGET_W
    # global H
    # global W
    # global D
    # global L

    #.... Initialize SORT .... 
        
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh) 
    #......................... 

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    count = 0
    
    for path, im, im0s, vid_cap, s in dataset:
        found = False
        count += 1
        #bev initial
        x_min = 1e09
        x_max = -1e09
        y_min = 1e09
        y_max = -1e09

        #cam_to_img = get_calibration_cam_to_image('/home/dzhura/Code/3D-BoundingBox/eval/calib/{f_name}'.format(f_name = p.stem), 2)
        im3d = im.copy()
        #if count > 10: break
        #if count < 336: continue
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        #pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # camera path
            cam_to_img = get_calibration_cam_to_image('/home/dzhura/Code/3D-BoundingBox/eval/calib/{f_name}.txt'.format(f_name = p.stem), 2)
            optic_center = ( im0.shape[1]//2, im0.shape[0]) #project_3d_pt((0, 0, 0), cam_to_img)

            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            im3d = im0.copy()
            im_origin = im0.copy()
            bev_3d = im0.copy()
            #plot_3d_pts(bev_3d, [(0, 0, 0)])



            #bev_3d, mat, inv_mat = ipm_from_opencv(im3d, cam_to_img) #src, dst)            

            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                #yolo_masks = reversed(masks)
                #yolo_masks = scale_masks(im.shape[2:], masks, im0.shape)

                # Segments
                if True:
                    segments = reversed(masks2segments(masks))
                    segments = [scale_segments(im.shape[2:], x, im0.shape).round() for x in segments]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(6), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors) #, 0.5)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                
                #im3d = annotator.im.copy()

                #yolo_mask = np.zeros(im[i].shape[2:], np.uint8)
                yolo_mask = torch.zeros_like(im[i])
                yolo_mask = plot_masks(yolo_mask, masks, mcolors) #, 0.5)
                yolo_mask = scale_masks(im.shape[2:], yolo_mask, im0.shape)
                #cv2.imshow('yolo_mask', yolo_mask)
                # Mask plotting ----------------------------------------------------------------------------------------

                if trk:
                    #Tracking ----------------------------------------------------
                    dets_to_sort = np.empty((0,6))
                    for x1,y1,x2,y2,conf,detclass in det[:, :6].cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, 
                                        np.array([x1, y1, x2, y2, 
                                                    conf, detclass])))

                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks =sort_tracker.getTrackers()

                    for track in tracks:
                        annotator.draw_trk(line_thickness,track)

                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        annotator.draw_id(bbox_xyxy, identities, categories, names)
            
                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    bbox = np.array(
                    [
                        [xyxy[0].item(), xyxy[1].item()],
                        [xyxy[2].item(), xyxy[3].item()],
                    ], np.int32 )
                    #print("bbox :", bbox)
                    #print("seg :", segments[j])
                    #x1, y1, x2, y2 = [int(i) for i in bbox]
                    # center = ( (xyxy[0].item() + xyxy[2].item())//2 , (xyxy[1].item() + xyxy[3].item())//2 )
                    # warp_center = warp_perspective(center, mat)
                    # cv2.circle(bev_3d, warp_center, 5, cv_colors.RED.value, thickness=-1)
                    # est_center = (warp_center[0]// 10, H , warp_center[1]// 10)

                    loc = None
                    lower_face = None
                    dim = get_dim_custom(cls)
                    print("dim :", dim)
                    
                    if dim is not None:
                        #angles = get_bottom_variants(bev_3d, bbox, mat, inv_mat, dim, thickness = 10)
                        #loc, lower_face, color = plot_regressed_3d_bbox_fast(im_origin, im3d, cam_to_img, cls, bbox, yolo_mask, optic_center, mat, inv_mat)
                        loc, lower_face, orient = plot_regressed_3d_bbox(im_origin, im3d, cam_to_img, cls, bbox, yolo_mask)
                    # if loc is not None:
                        # center = (loc[0], loc[1] , loc[2])
                        # loc_bev = warp_perspective(project_3d_pt(center, cam_to_img), mat)
                        # cv2.circle(bev_3d, (int(loc_bev[0]), int(loc_bev[1])), 3, cv_colors.RED.value, thickness=-1)
                    #bev_mask, mat, inv_mat = ipm_from_opencv(yolo_mask, src, dst)
                    #plot_regressed_3d_bbox_mod(im3d, cam_to_img, cls, bbox, bev_mask)
                    #est_loc = None #regress_location(im3d, cam_to_img, cls, bbox, yolo_mask, est_center)
                    # if loc is not None:
                        # print("loc :",  loc)
                        # print("est_center :", est_center)
                        # #print("est_loc :", est_loc)
                        # print("dist :", np.linalg.norm(np.array(loc) - np.array(est_center)))
                    # if lower_face is not None:
                        # found = True
                        # print("cls :", cls.item())
                        # #bev_pts = to_warp(lower_face, mat)
                        # for p in bev_pts:
                            # if x_min > p[0]: x_min = p[0]
                            # if x_max < p[0]: x_max = p[0]
                            # if y_min > p[1]: y_min = p[1]
                            # if y_max < p[1]: y_max = p[1]
                        # bev_pts = np.array(bev_pts, np.int32)
                        # print("dist 1 :", np.linalg.norm(np.array(bev_pts[0]) - np.array(bev_pts[1])))
                        # print("dist 2 :", np.linalg.norm(np.array(bev_pts[1]) - np.array(bev_pts[2])))

                        # bev_pts = bev_pts.reshape((-1, 1, 2))
                        # #cv2.polylines(bev_3d, [bev_pts], True, cv_colors.BLUE.value, 2)
                        
                        # rect = cv2.minAreaRect(bev_pts)
                        # box = cv2.boxPoints(rect)
                        # box = np.int0(box)
                        # cv2.drawContours(bev_3d,[box],0, color, 10)

                    #update BEV
                    # if x_min > loc[0]: x_min = loc[0]
                    # if y_min > loc[1]: y_min = loc[1]
                    # if x_max < loc[0]: x_max = loc[0]
                    # if y_max < loc[1]: y_max = loc[1]
                    # segj = segments[j].reshape(-1)
                    # cv2.drawContours(im0, [segj], -1, cv_colors.RED.value , 3)
                    if save_txt:  # Write to file
                        segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            #im0 = plot_regressed_3d_bbox_test(im0, cam_to_img, cls, box_2d, )
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im3d)
                #cv2.waitKey(1)  # 1 millisecond
                #cv2.waitKey(0)

            # Save results (image with detections)


            # r = 10
            # for alpha in range(0, 180, 10):

                # #rec = [r * np.cos(np.deg2rad(alpha)), 0, 0] #r * np.sin(np.deg2rad(alpha))]
                # rec = [0, 0, r * np.sin(np.deg2rad(alpha))]
                # plot_3d_pts(im3d, [rec], cam_to_img = cam_to_img)

            # rec.append(project_3d_pt((-w/2, y0, 0), cam_to_img))
            # rec.append(project_3d_pt((-w//2, y0 + h, 0), cam_to_img))
            # rec.append(project_3d_pt((w//2, y0 + h, 0), cam_to_img))
            # rec.append( project_3d_pt((w//2, y0, 0), cam_to_img))


            # Extract translation vector
            # t = cam_to_img[:, 3]
            # print(t)
            
            w = 20
            h = -3.4
            d = 20
            l = 40
            # [ -5.84265631,   3.99833006, 178.6208895 ]
            # alpha = np.radians(-5.84265631)
            
            # rec = []
            # rec.append([-w/2, h + d* np.sin(alpha) , d + 0])
            # rec.append([-w//2, h + (d+l)* np.sin(alpha), d + l])
            # rec.append([w//2, h + (d + l)* np.sin(alpha), d+l])
            # rec.append([w//2,  h + d* np.sin(alpha), d])
            
            # for p1 in rec:
                # for p2 in rec:
                    # cv2.line(im3d, project_3d_pt(p1, cam_to_img), project_3d_pt(p2, cam_to_img), cv_colors.MINT.value, 3)
                    
            # alpha = np.radians(0)
            
            # rec = []
            # rec.append([-w/2, h - d* np.sin(alpha) , d + 0])
            # rec.append([-w//2, h - (d+l)* np.sin(alpha), d + l])
            # rec.append([w//2, h - (d + l)* np.sin(alpha), d+l])
            # rec.append([w//2,  h - d* np.sin(alpha), d])
            
            # for p1 in rec:
                # for p2 in rec:
                    # cv2.line(im3d, project_3d_pt(p1, cam_to_img), project_3d_pt(p2, cam_to_img), cv_colors.RED.value, 3)
            #[, h, ]
            #plot_3d_pts(im3d, rec, cam_to_img = cam_to_img)
            # alpha = np.radians(0)
            # dim = [-h, 10, 10]
            # loc = [0,h//2 - 0.5 + 14* np.sin(alpha),14]
            # corners = get_3d_box(im3d, cam_to_img, 0, dim, loc, thickness = 3, color = cv_colors.GREEN.value)
            
            # dim = [-h, 10, 10]
            # loc = [0,h//2 + 14* np.sin(alpha),10]
            # corners = get_3d_box(im3d, cam_to_img, np.radians(3), dim, loc, thickness = 3, color = cv_colors.BLUE.value)

            # cv2.line(im3d, corners[0], corners[3], cv_colors.RED.value, 3)

            # center = [0, 1.820, 10]
            # plot_3d_pts(im3d, [center], center = center, cam_to_img = cam_to_img, color = cv_colors.BLUE.value)
            # center = [0, 0, 10]
            # plot_3d_pts(im3d, [center], center = center, cam_to_img = cam_to_img,  color = cv_colors.GREEN.value)
            # center = [0, 0, 10]
            # plot_3d_pts(im3d, [center], center = center, cam_to_img = cam_to_img,  color = cv_colors.YELLOW.value)
            # center = [0, 0, 15]
            # plot_3d_pts(im3d, [center], center = center, cam_to_img = cam_to_img,  color = cv_colors.PURPLE.value)
            # #plot_3d_pts(im3d, rec, center = center, cam_to_img = cam_to_img)


                    
                    
            # corners = [#[3.5205154964354195, 1.8978872704889223, 12.47321392411095],
                        # #[1.877086349302572, 1.8978872704889223, 12.530603734533997], 
                        # [3.5205154964354195, 0.3762958004889224, 12.47321392411095],
                        # [1.877086349302572, 0.3762958004889224, 12.530603734533997],
                        # #[3.655162528615822, 1.8978872704889223, 16.32900044142185],
                        # #[2.0117333814829745, 1.8978872704889223, 16.386390251844897],
                        # [3.655162528615822, 0.3762958004889224, 16.32900044142185], [2.0117333814829745, 0.3762958004889224, 16.386390251844897]]
                        
            # for p1 in corners:
                # for p2 in corners:
                    # cv2.line(im3d, project_3d_pt(p1, cam_to_img), project_3d_pt(p2, cam_to_img), cv_colors.MINT.value, 3)
   
            # save 3D
            # pad = 10
            # if found:
                # bev_3d = bev_3d[y_min - pad: y_max + pad, x_min - pad:x_max + pad]
            # # else:
                # # bev_3d = bev_3d
            # #bev_out = bev_3d[2*TARGET_H - P :6*TARGET_H + P//2, TARGET_W: 4*TARGET_W]
            # #v_comb = cv2.vconcat([frame_debug, rgb_out])
            #bev_3d =  bev_3d[0 : TARGET_H, 0: TARGET_W]
            # try:
                # imgs_comb = hconcat_resize([im3d, bev_3d])#, debug_bev])
                # im0 = imgs_comb
            # except:
                # im0 = bev_3d
            im0 = im3d
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--trk', action='store_true', help='Apply Sort Tracking')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
