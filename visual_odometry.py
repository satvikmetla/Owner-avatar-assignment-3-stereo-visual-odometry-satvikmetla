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
    sad_window = 5
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
    disp_left = np.where(disp_left <= 0, 1e-6, disp_left)
    fx = k_left[0, 0]

    if rectified:
        baseline = abs(t_left[0] - t_right[0])
    else:
        baseline = np.linalg.norm(t_left - t_right)

    depth_map = (fx * baseline) / disp_left

    return depth_map

def extract_features(image, detector='sift', mask=None):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    if detector.lower() == 'sift':
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(image, mask)
        
    elif detector.lower() == 'orb':
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(image, mask)
        
    else:
        raise ValueError(f"Unsupported detector: {detector}. Choose 'sift' or 'orb'")
    
    return kp, des

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
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf_matcher.knnMatch(des1, des2, k=k)
    if sort and k == 1:
        matches = sorted(matches, key=lambda x: x[0].distance)
    elif sort:
        matches = sorted(matches, key=lambda x: [m.distance for m in x])

    return matches

def filter_matches_distance(matches, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_matches = []
    if len(matches) > 0 and isinstance(matches[0], cv2.DMatch):
        filtered_matches = [m for m in matches if m.distance < dist_threshold]
    else:
        for m in matches:
            if len(m) >= 2:
                if m[0].distance < dist_threshold * m[1].distance:
                    filtered_matches.append(m[0])
    
    return filtered_matches


def get_3d_from_matches(filtered_matches, depth_map ,k_l_start, k_l_end, k):
    """
    Arguments:
    filtered_matches -- list of filtered matches from the pair of images
    depth_map -- Depth map constructed using 2 images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera intrinsic calibration matrix

    Returns:
    3d_points -- 3D positions of the matched features relative to the camera
    2d_points -- corresponding 2D points of next image
    """
    k_inv = np.linalg.inv(k)
    points_3d = []
    for match in filtered_matches:
        u, v = k_l_start[match.queryIdx].pt
        u, v = int(round(u)), int(round(v))

        if v >= depth_map.shape[0] or u >= depth_map.shape[1]:
            continue
        depth = depth_map[v, u]
        if depth <= 0 or np.isnan(depth):
            continue
        pixel = np.array([u, v, 1.0])
        point_3d = depth * (k_inv @ pixel)
        points_3d.append(point_3d)

    points_3d = np.array(points_3d, dtype=np.float32)

    return points_3d


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
    image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in match])
    if depth1 is None:
        E, mask = cv2.findEssentialMat(image1_points, image2_points,k,method=cv2.RANSAC,prob=0.999,threshold=1.0)
        _, rmat, tvec, _ = cv2.recoverPose(E, image1_points, image2_points,k)
    else:
        object_points = get_3d_from_matches(match, depth1, kp1, kp2, k)
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image2_points, k, None,
            flags=cv2.SOLVEPNP_ITERATIVE,iterationsCount=100,reprojectionError=3.0,confidence=0.99
        )
        rmat, _ = cv2.Rodrigues(rvec)

    return rmat, tvec, image1_points, image2_points


def quaternion_to_rotation_matrix(qx, qy, qz, qw):

    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Calculate rotation matrix elements
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

def rotation_matrix_to_quaternion(R):
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt  
    
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0: 
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else: 
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    return np.array([qx / norm, qy / norm, qz / norm, qw / norm])


def get_true_pose(file_path,frame_start,frame_end):
    gt_values = []


    # Open the ground truth pose file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line_num,line in enumerate(lines):
            if line_num < frame_start:
                continue
            if line_num > frame_end:
                break
            values = list(map(float, line.split()))
            tx = values[3]
            tz = values[11]
            gt_values.append([tx, tz])
    return gt_values

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
    PO = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00 ],
          [0.000000000000e+00, 7.188560000000e+02 ,1.852157000000e+02, 0.000000000000e+00],
          [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]
        ])
    P1 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02 ],
          [0.000000000000e+00, 7.188560000000e+02 ,1.852157000000e+02, 0.000000000000e+00],
          [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]
        ])

    images_path = config['images'] # Path to the images
    sequence = config['sequence'] # Sequence name
    frame_start = config['frame_start'] # This is the frame number to start with
    frame_end = frame_start + 1 # We are using only two frames

    images_0 = os.path.join(images_path, 'dataset', 'sequences', f"{sequence:02d}", 'image_0')
    images_1 = os.path.join(images_path, 'dataset', 'sequences', f"{sequence:02d}", 'image_1')

    image_l_start_file = f"{images_0}/{frame_start:06d}.png"
    image_l_end_file = f"{images_0}/{frame_end:06d}.png"
    image_r_start_file = f"{images_1}/{frame_start:06d}.png"
    image_r_end_file = f"{images_1}/{frame_end:06d}.png"

    # Load the images and poses for the two frames using the correct sequence and frame numbers
    image_l_start = cv2.imread(image_l_start_file, cv2.IMREAD_GRAYSCALE)
    image_r_start = cv2.imread(image_r_start_file, cv2.IMREAD_GRAYSCALE)
    image_l_end = cv2.imread(image_l_end_file, cv2.IMREAD_GRAYSCALE)
    image_r_end = cv2.imread(image_r_end_file, cv2.IMREAD_GRAYSCALE)
    trajectory = [[0,0,0,0,0,0,0,1]] # Initial pose


    # Decompose projection matrices 
    k0,r0,t0 = decompose_projection_matrix(PO)
    k1,r1,t1 = decompose_projection_matrix(P1)

    # Compute disparity map for start frames
    disp_left_map = compute_left_disparity_map(image_l_start, image_r_start, matcher='bm', rgb=False, verbose=False)

    # Compute depth map for start frames
    depth_map = calc_depth_map(disp_left_map, k0, t0, t1, rectified=True)


    # Extract features from start left frame
    kp_l_start, des_l_start = extract_features(image_l_start)
    kp_l_end,des_l_end = extract_features(image_l_end)

    # Track features from start left frame to end left frame
    matches = match_features(des_l_start, des_l_end)
    filtered_matches = filter_matches_distance(matches, 0.75)

    # Get 3D points of good matches
    # object_points = get_3d_from_matches(filtered_matches, depth_map, kp_l_start, kp_l_end, k0)

    # Estimate motion from 3D points

    rmat, tvec, image1_points, image2_points = estimate_motion(filtered_matches, kp_l_start, kp_l_end, k0, depth_map)

    # Get pose of end frame and append to trajectory
    prev_pose = trajectory[-1]
    prev_index, prev_x, prev_y, prev_z, prev_qx, prev_qy, prev_qz, prev_qw = prev_pose
    prev_rmat = quaternion_to_rotation_matrix(prev_qx, prev_qy, prev_qz, prev_qw)
    prev_pos = np.array([prev_x, prev_y, prev_z])
    new_pos = prev_rmat @ tvec.flatten() + prev_pos
    new_rmat = prev_rmat @ rmat
    new_quat = rotation_matrix_to_quaternion(new_rmat)
    
    # Create new pose entry
    new_pose = [
        prev_index + 1,
        float(new_pos[0]),
        float(new_pos[1]),
        float(new_pos[2]),
        float(new_quat[0]),
        float(new_quat[1]),
        float(new_quat[2]),
        float(new_quat[3])
    ]
    
    # Append to trajectory
    trajectory.append(new_pose)

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

    # Camera projection matrices (hard code it since it doesn't change)
    PO = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00 ],
          [0.000000000000e+00, 7.188560000000e+02 ,1.852157000000e+02, 0.000000000000e+00],
          [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]
        ])
    P1 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02 ],
          [0.000000000000e+00, 7.188560000000e+02 ,1.852157000000e+02, 0.000000000000e+00],
          [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]
        ])
    
    # Decompose projection matrices 
    k0,r0,t0 = decompose_projection_matrix(PO)
    k1,r1,t1 = decompose_projection_matrix(P1)
    
    images_l = os.path.join(images_path, 'dataset', 'sequences', f"{sequence:02d}", 'image_0')
    images_r = os.path.join(images_path, 'dataset', 'sequences', f"{sequence:02d}", 'image_1')

    # SVO Loop between frame_start and frame_end
    for i in range(frame_start, frame_end):
        image_l_start_file = f"{images_l}/{i:06d}.png"
        image_l_end_file = f"{images_l}/{i+1:06d}.png"
        image_r_start_file = f"{images_r}/{i:06d}.png"
        image_r_end_file = f"{images_r}/{i+1:06d}.png"

        # Load the images and poses for the two frames using the correct sequence and frame numbers
        image_l_start = cv2.imread(image_l_start_file, cv2.IMREAD_GRAYSCALE)
        image_r_start = cv2.imread(image_r_start_file, cv2.IMREAD_GRAYSCALE)
        image_l_end = cv2.imread(image_l_end_file, cv2.IMREAD_GRAYSCALE)
        image_r_end = cv2.imread(image_r_end_file, cv2.IMREAD_GRAYSCALE)
        disp_left_map = compute_left_disparity_map(image_l_start, image_r_start, matcher='bm', rgb=False, verbose=False)

        # Compute depth map for start frames
        depth_map = calc_depth_map(disp_left_map, k0, t0, t1, rectified=True)


        # Extract features from start left frame
        kp_l_start, des_l_start = extract_features(image_l_start)
        kp_l_end,des_l_end = extract_features(image_l_end)

        # Track features from start left frame to end left frame
        matches = match_features(des_l_start, des_l_end)
        filtered_matches = filter_matches_distance(matches, 0.75)

        # Get 3D points of good matches
        # object_points = get_3d_from_matches(filtered_matches, depth_map, kp_l_start, kp_l_end, k0)

        # Estimate motion from 3D points

        rmat, tvec, image1_points, image2_points = estimate_motion(filtered_matches, kp_l_start, kp_l_end, k0, depth_map)

        # Get pose of end frame and append to trajectory
        prev_pose = trajectory[-1]
        prev_index, prev_x, prev_y, prev_z, prev_qx, prev_qy, prev_qz, prev_qw = prev_pose
        prev_rmat = quaternion_to_rotation_matrix(prev_qx, prev_qy, prev_qz, prev_qw)
        prev_pos = np.array([prev_x, prev_y, prev_z])
        new_pos = prev_rmat @ tvec.flatten() + prev_pos
        new_rmat = prev_rmat @ rmat
        new_quat = rotation_matrix_to_quaternion(new_rmat)
        
        # Create new pose entry
        new_pose = [
            prev_index + 1,
            float(new_pos[0]),
            float(new_pos[1]),
            float(new_pos[2]),
            float(new_quat[0]),
            float(new_quat[1]),
            float(new_quat[2]),
            float(new_quat[3])
        ]
        
        # Append to trajectory
        trajectory.append(new_pose)

    poses_file_path = os.path.join(poses_path, 'dataset', 'poses', f"{sequence:02d}.txt" )
    gt_poses = get_true_pose(poses_file_path,frame_start,frame_end)

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

