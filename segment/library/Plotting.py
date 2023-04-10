import cv2
import numpy as np
from enum import Enum
import itertools
import random


from .File import *
from .Math import *

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)
    WHITE = (255,255,255)
    BLACK = (0, 0, 0)

def constraint_to_color(constraint_idx):
    return {
        0 : cv_colors.PURPLE.value, #left
        1 : cv_colors.ORANGE.value, #top
        2 : cv_colors.MINT.value, #right
        3 : cv_colors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img): #, calib_file=None):
    # if calib_file is not None:
        # cam_to_img = get_calibration_cam_to_image(calib_file)
        # R0_rect = get_R0(calib_file)
        # Tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    #point = np.dot(np.dot(cam_to_img, R0_rect), point)
    #point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)

    return point


# take in 3d points and plot them on image as red circles
def plot_3d_pts(img, pts, center=None, calib_file=None, cam_to_img=None, relative=False, constraint_idx=None, color = cv_colors.RED.value):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
    #pts = pts[:4]

    for pt in pts:
        if relative:
            pt = [i + center[j] for j,i in enumerate(pt)] # more pythonic

        point = project_3d_pt(pt, cam_to_img)
        print(point)

        #color = cv_colors.RED.value

        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)

        cv2.circle(img, (point[0], point[1]), 5, color, thickness=-1)


def get_3d_mask(mask, cam_to_img, ry, dimension, center): #, cls = None, vertices = None, triangles = None):


    R = rotation_matrix(ry)
    corners = create_corners(dimension, location=center, R=R)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    hull = cv2.convexHull(np.int32(np.array(box_3d)))

    cv2.fillConvexPoly(mask, hull, cv_colors.WHITE.value)

    # else:
        # Rt = np.eye(4)
        # #t = np.array([x, y, z])
        # Rt[:3, 3] = t
        # Rt[:3, :3] = rotation_matrix(orient + np.pi/2) #euler_to_Rot(yaw, pitch, roll).T
        # #Rt = Rt[:3, :]
        # P = np.ones((vertices.shape[0],vertices.shape[1]+1))
        # P[:, :-1] = vertices
        # P = P.T
        # #print(cam_to_img)
        # img_cor_points = np.dot(cam_to_img, np.dot(Rt, P))
        # img_cor_points = img_cor_points.T
        # img_cor_points[:, 0] /= img_cor_points[:, 2]
        # img_cor_points[:, 1] /= img_cor_points[:, 2]
        # draw_obj(overlay, img_cor_points, triangles, color)
        
        # for t in triangles:
            # coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)

        # cv2.fillConvexPoly(image, coord, cv_colors.WHITE.value)
        #cv2.polylines(image, np.int32([coord]), 1, cv_colors.WHITE.value)




def plot_3d_box(img, cam_to_img, ry, dimension, center, color = None, thickness = 4):
    print("location :", center)
    #thickness = 2
    if color is None:
        color = list(np.random.random(size=3) * 256)
    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    R = rotation_matrix(ry)

    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    #plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False, constraint_idx = 3)

    box_3d = []
    #print("corners :", corners)
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    # hull = cv2.convexHull(np.int32(np.array(box_3d)))
    # cv2.fillConvexPoly(img, hull, color=cv_colors.RED.value)
    # cv2.circle(img, (box_3d[0][0], box_3d[0][1]), 5, cv_colors.RED.value, thickness=-1)
    # cv2.circle(img, (box_3d[1][0], box_3d[1][1]), 5, cv_colors.RED.value, thickness=-1)
    # cv2.circle(img, (box_3d[4][0], box_3d[4][1]), 5, cv_colors.RED.value, thickness=-1)
    # cv2.circle(img, (box_3d[5][0], box_3d[5][1]), 5, cv_colors.RED.value, thickness=-1)
    
    #indexes = [0,1, 4, 5]
 
    # cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[1][0],box_3d[1][1]), cv_colors.PURPLE.value, thickness)
    # cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.PURPLE.value, thickness)
    # cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.PURPLE.value, thickness)
    # cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.PURPLE.value, thickness)

    #TODO put into loop
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), color, thickness)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), color, thickness)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), color, thickness)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), color, thickness)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), color, thickness)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), color, thickness)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), color, thickness)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), color, thickness)

    for i in range(0,7,2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), color, thickness)

    # front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]

    # cv2.line(img, front_mark[0], front_mark[3], cv_colors.BLUE.value, 1)
    # cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 1)
    return [box_3d[0],box_3d[1], box_3d[5], box_3d[4]]

def plot_2d_box(img, box_2d, thickness = 2):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, thickness)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, thickness)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, thickness)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, thickness)


def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
    return image
