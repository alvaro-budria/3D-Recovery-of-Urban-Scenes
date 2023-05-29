import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2
import math
import sys
import random
from typing import Tuple


def line_draw(line, canv, size):
    def get_y(t):
        return -(line[0] * t + line[2]) / line[1]

    def get_x(t):
        return -(line[1] * t + line[2]) / line[0]

    w, h = size

    if line[0] != 0 and abs(get_x(0) - get_x(w)) < w:
        beg = (get_x(0), 0)
        end = (get_x(h), h)
    else:
        beg = (0, get_y(0))
        end = (w, get_y(w))
    canv.line([beg, end], width=4)


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def normalise_last_coord(x):
    xn = x  / x[2,:]
    return xn


def normalize_points(points):
    """
    Normalize the points to have zero mean and unit standard deviation.

    Args:
        points: 3xN matrix of homogeneous coordinates of the points

    Returns:
        points_norm: 3xN matrix of normalized homogeneous coordinates of the points
        T: 3x3 matrix of the transformation
    """
    Ncoords, Npts = points.shape
    mean = np.mean(points, axis=1)
    d_centroid = points[:, :] - mean[:, np.newaxis]
    s = np.sqrt(2) / np.mean(np.sqrt(np.sum(d_centroid ** 2, axis=0)))
    T = np.eye(Ncoords)
    T[0, 0] = s
    T[1, 1] = s
    T[0, 2] = -mean[0] * s
    T[1, 2] = -mean[1] * s
    points_norm = T @ points

    return points_norm, T


def fundamental_matrix(P1: np.ndarray, P2: np.ndarray):
    """
    Computes the fundamental matrix from corresponding points (x, y) in each image.

    1. Normalize the image points so that the origin is at centroid and mean distance from origin is sqrt(2).
    2. Compute the matrix W for which Wf = 0 by using the normalized points.
    3. Compute the SVD of W and extracts the fundamental matrix f from the last column of V corresponding to the smallest singular value.
    4. Compose the fundamental matrix F from f.
    5. Compute the SVD of F and enforce the rank 2 constraint on F by zeroing out the smallest singular value.
    6. Recompose F from the modified singular values.
    7. Denormalize F.

    Args:
        P1: 3xN matrix of homogeneous coordinates of the points in the first image
        P2: 3xN matrix of homogeneous coordinates of the points in the second image
    """
    P1_norm, T1 = normalize_points(P1)
    P2_norm, T2 = normalize_points(P2)

    P1_norm = normalise_last_coord(P1_norm)
    P2_norm = normalise_last_coord(P2_norm)

    Ncoords, Npts = P1_norm.shape
    W = np.zeros((Npts, 9))

    for i in range(Npts):
        u1, v1, _ = P1_norm[:, i]
        u2, v2, _ = P2_norm[:, i]
        W[i, :] = [u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, 1]

    U, S, Vt = LA.svd(W)
    f = Vt[-1, :]
    F = f.reshape((3, 3))

    U, S, Vt = LA.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    F = T2.T @ F @ T1
    return F


def essential_matrix(t, R):
    """
    Computes the essential matrix from the camera pose.

    Args:
        t: 3x1 translation vector
        R: 3x3 rotation matrix

    Returns:
        E: 3x3 essential matrix
    """
    t = t[0]
    T = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
        ])
    E = T @ R
    return E


def kp_match(img1, img2, descr='orb', dist_th=0.85):
    # Initiate ORB detector
    orb = cv2.ORB_create(3000)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Keypoint matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    return kp1, kp2, matches


def draw_matches(img1, kp1, img2, kp2, matches):
    # Show "good" matches
    img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()


def compute_inliers(F, P1, P2, th):
    """
    Compute the number of inliers consistent with F by the number of 
    correspondences for which d < th pixels. We use the sampson distance.

    Args:
        F: 3x3 matrix of the homography
        P1: 3xN matrix of homogeneous coordinates of the points in the first image
        P2: 3xN matrix of homogeneous coordinates of the points in the second image
        th: threshold for the distance between the points

    Returns:
        idx: indices of the inliers
    """
    Ncoords, Npts = P1.shape
    d = np.empty(Npts)

    normalized_P1 = normalise_last_coord(P1)
    normalized_P2 = normalise_last_coord(P2)

    for i in range(Npts):
        x1 = normalized_P1[:, i]
        x2 = normalized_P2[:, i]
        
        Fx1 = F @ x1
        Ftx2 = F.T @ x2

        d[i] = (x2.T @ F @ x1)**2 / (Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2)

    idx = np.where(d < th)[0]
    geometric_error = np.sum(d[idx])
    return idx, geometric_error


def ransac_fundamental_matrix(P1, P2, th, max_it_0):
    """
    RANSAC algorithm to estimate the fundamental matrix
    """
    Ncoords, Npts = P1.shape

    it = 0
    best_geometric_error = np.inf
    max_it = max_it_0
    best_inliers = np.empty(1)

    while it < max_it:
        indices = random.sample(range(1, Npts), 8)
        F = fundamental_matrix(P1[:,indices], P2[:,indices])
        inliers, geometric_error = compute_inliers(F, P1, P2, th)

        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
            best_geometric_error = geometric_error
 
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 - fracinliers**4
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)

        it += 1

        # To avoid an infinite loop
        max_it_0 = min(max_it,max_it_0)

    # compute H from all the inliers
    F = fundamental_matrix(P1[:, best_inliers], P2[:, best_inliers])

    return F, best_inliers, best_geometric_error


def compute_epipolar_lines(F, points):
    return F @ points


def draw_epipolar_lines(img_path: str, l: np.ndarray, points: np.ndarray, inliers: Tuple[int, int, int]) -> np.ndarray:
    m1, m2, m3 = inliers
    I = Image.open(img_path)
    size = I.size
    canv = ImageDraw.Draw(I)
    line_draw(l[:,m1], canv, size)
    line_draw(l[:,m2], canv, size)
    line_draw(l[:,m3], canv, size)
    canv.ellipse((round(points[0,m1]), round(points[1,m1]), round(points[0,m1])+7, round(points[1,m1])+7), fill = 'red', outline ='red')
    canv.ellipse((round(points[0,m2]), round(points[1,m2]), round(points[0,m2])+7, round(points[1,m2])+7), fill = 'red', outline ='red')
    canv.ellipse((round(points[0,m3]), round(points[1,m3]), round(points[0,m3])+7, round(points[1,m3])+7), fill = 'red', outline ='red')
    return I