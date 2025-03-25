import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import pandas as pd
import argparse
import yaml
import datetime

def decompose_projection_matrix(p):
    '''
    Shortcut to use cv2.decomposeProjectionMatrix(), which only returns k, r, t, and divides
    t by the scale, then returns it as a vector with shape (3,) (non-homogeneous)
    
    Arguments:
    p -- projection matrix to be decomposed
    
    Returns:
    k, r, t -- intrinsic matrix, rotation matrix, and 3D translation vector
    
    '''

    # Free implementation provided! You may use this function as is.
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    
    return k, r, t

def compute_left_disparity_map(img_left, img_right, matcher='bm', rgb=False, verbose=False):
    '''
    Takes a left and right stereo pair of images and computes the disparity map for the left
    image. Pass rgb=True if the images are RGB.
    
    Arguments:
    img_left -- image from left camera
    img_right -- image from right camera
    
    Optional Arguments:
    matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM matching
    rgb -- (bool) set to True if passing RGB images as input
    verbose -- (bool) set to True to report matching type and time to compute
    
    Returns:
    disp_left -- disparity map for the left camera image
    
    '''
    # Free implementation provided! You may use this function as is.

    # Feel free to read OpenCV documentation and tweak these values. These work well
    sad_window = 6
    num_disparities = sad_window*16
    block_size = 11
    matcher_name = matcher
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size
                                     )
        
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 3 * sad_window ** 2,
                                        P2 = 32 * 3 * sad_window ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                       )
    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    start = datetime.datetime.now()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = datetime.datetime.now()
    if verbose:
        print(f'Time to compute disparity map using Stereo{matcher_name.upper()}:', end-start)
    
    return disp_left


def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):
    '''
    Calculate depth map using a disparity map, intrinsic camera matrix, and translation vectors
    from camera extrinsic matrices (to calculate baseline). Note that default behavior is for
    rectified projection matrix for right camera. If using a regular projection matrix, pass
    rectified=False to avoid issues.
    
    Arguments:
    disp_left -- disparity map of left camera
    k_left -- intrinsic matrix for left camera
    t_left -- translation vector for left camera
    t_right -- translation vector for right camera
    
    Optional Arguments:
    rectified -- (bool) set to False if t_right is not from rectified projection matrix
    
    Returns:
    depth_map -- calculated depth map for left camera
    
    '''
    pass

def stereo_2_depth(img_left, img_right, P0, P1, matcher='bm', rgb=False, verbose=False, 
                   rectified=True):
    '''
    Takes stereo pair of images and returns a depth map for the left camera. If your projection
    matrices are not rectified, set rectified=False.
    
    Arguments:
    img_left -- image of left camera
    img_right -- image of right camera
    P0 -- Projection matrix for the left camera
    P1 -- Projection matrix for the right camera
    
    Optional Arguments:
    matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM
    rgb -- (bool) set to True if images passed are RGB. Default is False
    verbose -- (bool) set to True to report computation time and method
    rectified -- (bool) set to False if P1 not rectified to P0. Default is True
    
    Returns:
    depth -- depth map for left camera
    
    '''
    pass


def extract_features(image, detector='sift', mask=None):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    pass

def match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2):
    """
    Match features from two images
    
    Note : you need not implement both Brute Force and FLANN matching. Choose one of them. Although, it is recommended to learn using both.

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image
    matching -- (str) can be 'BF' for Brute Force or 'FLANN'
    detector -- (str) can be 'sift or 'orb'. Default is 'sift'
    sort -- (bool) whether to sort matches by distance. Default is True
    k -- (int) number of neighbors to match to each feature.

    Returns:
    matches -- list of matched features from two images. Each match[i] is k or less matches for 
               the same query descriptor
    """
    pass

def filter_matches_distance(matches, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    pass


def estimate_motion(match, kp1, kp2, k, depth1=None, max_depth=3000):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera intrinsic calibration matrix 
    
    Optional arguments:
    depth1 -- Depth map of the first frame. Set to None to use Essential Matrix decomposition
    max_depth -- Threshold of depth to ignore matched features. 3000 is default

    Returns:
    rmat -- estimated 3x3 rotation matrix
    tvec -- estimated 3x1 translation vector
    image1_points -- matched feature pixel coordinates in the first image. 
                     image1_points[i] = [u, v] -> pixel coordinates of i-th match
    image2_points -- matched feature pixel coordinates in the second image. 
                     image2_points[i] = [u, v] -> pixel coordinates of i-th match
               
    """

    pass



def get_true_pose(file_path):
    pass

def get_camera_matrix(config):
    pass

def save_trajectory_to_csv(trajectory, file_name):
    df = pd.DataFrame(trajectory, columns=['index', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    df.to_csv(file_name, index=False)


def two_frame_vo(config_path):
    """Main function for stereo visual odometry using only two frames."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Camera projection matrices (hard code it since it doesn't change)
    P0 = None
    P1 = None

    images_path = config['images'] # Path to the images
    sequence = config['sequence'] # Sequence name
    frame_start = config['frame_start'] # This is the frame number to start with
    frame_end = frame_start + 1 # We are using only two frames

    # Load the images and poses for the two frames using the correct sequence and frame numbers
    image_l_start = None
    image_r_start = None
    image_l_end = None
    image_r_end = None
    trajectory = [[0,0,0,0,0,0,0,1]] # Initial pose


    # Decompose projection matrices 

    # Compute disparity map for start frames

    # Compute depth map for start frames

    # Extract features from start left frame

    # Track features from start left frame to end left frame

    # Get 3D points of good matches

    # Estimate motion from 3D points

    # Get pose of end frame and append to trajectory

    for t in trajectory:
        print(t)



def full_svo(config_path):
    """Main function for stereo visual odometry."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    images_path = config['images'] # Path to the images
    poses_path = config['poses'] # Path to the ground truth poses
    sequence = config['sequence'] # Sequence name
    frame_start = config['frame_start'] # This is the frame number to start with
    frame_end = config['frame_end'] # This is the frame number to end with
    csv_path = config.get('csv_path', './output.csv')

    # Iterate over images, poses and compute visual odometry
    trajectory = [[0,0,0,0,0,0,0,1]] # Initial pose

    # SVO Loop between frame_start and frame_end

    for t in trajectory:
        print(t)

    # Save trajectory to csv
    save_trajectory_to_csv(trajectory, csv_path)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Odometry App")
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    parser.add_argument('--simple', action='store_true', help='Use only two frames')
    args = parser.parse_args()
    if args.simple:
        two_frame_vo(args.config)
    else:
        full_svo(args.config)

