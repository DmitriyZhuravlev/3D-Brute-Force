"""
Functions to read from files
TODO: move the functions that read label from Dataset into here
"""
import numpy as np
import cv2


# def get_calibration_cam_to_image(cab_f, num):
    # for line in open(cab_f):
        # if 'P' + str(num) +':' in line:
        # #if 'P2:' in line:
            # cam_to_img = line.strip().split(' ')
            # cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            # cam_to_img = np.reshape(cam_to_img, (3, 4))
            # #y_image = P2 * R0_rect * R0_rot * x_ref_coord
            # return cam_to_img

    # file_not_found(cab_f)

# def get_R0_rot(cab_f, num):
    # for line in open(cab_f):
        # if 'P' + str(num) +':' in line:
        # #if 'P2:' in line:
            # cam_to_img = line.strip().split(' ')
            # cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            # cam_to_img = np.reshape(cam_to_img, (3, 4))
            # #return cam_to_img
            # R0_rect = get_R0(cab_f)

            # #y_image = P2 * R0_rect * R0_rot * x_ref_coord
            # return np.dot(cam_to_img, R0_rect)

    # file_not_found(cab_f)
    
def get_calibration_cam_to_image_02(cab_f, num):
    #print("calib file: ", cab_f, num)
    for line in open(cab_f):
        if 'P' + str(num) +':' in line:
        #if 'P2:' in line:
            P2 = line.strip().split(' ')
            P2 = np.asarray([float(number) for number in P2[1:]])
            P2 = np.reshape(P2, (3, 4))
        # if 'R0_rect' in line:
            # R0 = line.strip().split(' ')
            # R0 = np.asarray([float(number) for number in R0[1:]])
            # R0 = np.reshape(R0, (3, 3))
            
            # R0_rect = np.zeros([4,4])
            # R0_rect[3,3] = 1
            # R0_rect[:3,:3] = R0
            #return cam_to_img
            #R0_rect = get_R0(cab_f)

            #y_image = P2 * R0_rect * R0_rot * x_ref_coord
            #print("calibration_cam_to_image :", np.dot(P2, R0_rect))
            return P2 #np.dot(P2, R0_rect)

    file_not_found(cab_f)
    
def get_calibration_cam_to_image_o(cab_f, num):
    f = 1542.303223
    #f = 1242.303223
    K = np.array([[f, 0.000000, 961.741882, 0],
                            [0.000000, f, 527.682922, 0],
                            [0.000000, 0.000000, 1.000000, 1]])
                            
    R = np.identity(4, dtype=np.float32)
    R[3,3] = 3.4
    #[ -5.84265631,   3.99833006, 178.6208895 ]
    R[0:3, 0:3] = cv2.Rodrigues((np.radians(-584), np.radians(3.998), np.radians(178.62)))[0]
    cam_to_img = np.matmul(K , R)
    print(cam_to_img)
    return cam_to_img
    
    
def get_calibration_cam_to_image(cab_f, num):
    
    MAT1 =  [[-1.60525409e+03, -1.22393615e+02,  8.43697686e+02,
         2.01907948e+03],
       [-9.03665856e-01, -1.58765090e+03,  3.69475732e+02,
         3.92055420e+02],
       [-6.98029946e-02, -1.01378446e-01,  9.92396066e-01,
         1.04970144e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]
         


    cam_to_img = np.array( MAT1, dtype=np.float32)
    return cam_to_img


def get_P(cab_f, num):
    for line in open(cab_f):
        if ('P_rect_0'+ str(num)) in line:
        #if 'P_rect_02' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            return_matrix = np.zeros((3,4))
            return_matrix = cam_P.reshape((3,4))
            return return_matrix

    # try other type of file
    return get_calibration_cam_to_image

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))

            R0_rect = np.zeros([4,4])
            R0_rect[3,3] = 1
            R0_rect[:3,:3] = R0

            return R0_rect

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (3, 4))

            Tr_to_velo = np.zeros([4,4])
            Tr_to_velo[3,3] = 1
            Tr_to_velo[:3,:4] = Tr

            return Tr_to_velo

def file_not_found(filename):
    print("\nError! Can't read calibration file, does %s exist?"%filename)
    exit()
