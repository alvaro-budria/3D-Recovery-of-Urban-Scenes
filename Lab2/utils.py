import sys
import math
import random
import plotly.graph_objects as go
import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import Tuple
from math import ceil
from scipy.ndimage import map_coordinates


def draw_matches(img1, kp1, img2, kp2, matches):
    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img)
    plt.show()


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)
    fig = plt.gcf()
    fig.set_size_inches(16, 8)


def get_transformed_pixels_coords(img, H, shift=None):
    ys, xs = np.indices(img.shape[:2]).astype("float64")
    if shift is not None:
        ys += shift[1]
        xs += shift[0]
    ones = np.ones(img.shape[:2])
    coords = np.stack((xs, ys, ones), axis=2)
    coords_H = (H @ coords.reshape(-1, 3).T).T
    coords_H /= coords_H[:, 2, np.newaxis]
    cart_H = coords_H[:, :2]

    return cart_H.reshape((*img.shape[:2], 2))


def apply_H_fixed_image_size(img, H, corners):
    h, w = img.shape[:2]  # these two quantities are swapped when we convert to np.array

    # define corners
    C1 = np.array([1, 1, 1])
    C2 = np.array([w, 1, 1])
    C3 = np.array([1, h, 1])
    C4 = np.array([w, h, 1])

    # transform corners
    HC1 = H @ C1
    HC2 = H @ C2
    HC3 = H @ C3
    HC4 = H @ C4
    HC1 = HC1 / HC1[2]
    HC2 = HC2 / HC2[2]
    HC3 = HC3 / HC3[2]
    HC4 = HC4 / HC4[2]

    xmin, xmax  = corners[0], corners[1]
    ymin, ymax = corners[2], corners[3]
    S_x = ceil(xmax - xmin + 1)
    S_y = ceil(ymax - ymin + 1)

    # transform image
    H_inv = np.linalg.inv(H)

    out = np.zeros((S_y, S_x, 3))
    shift = (xmin, ymin)
    interpolation_coords = get_transformed_pixels_coords(out, H_inv, shift=shift)
    interpolation_coords[:, :, [0, 1]] = interpolation_coords[:, :, [1, 0]]
    interpolation_coords = np.swapaxes(np.swapaxes(interpolation_coords, 0, 2), 1, 2)

    out[:, :, 0] = map_coordinates(img[:, :, 0], interpolation_coords)
    out[:, :, 1] = map_coordinates(img[:, :, 1], interpolation_coords)
    out[:, :, 2] = map_coordinates(img[:, :, 2], interpolation_coords)

    return out.astype("uint8")


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
    d_centroid = points[:2, :] - mean[:2, np.newaxis]
    s = np.sqrt(2) / np.mean(np.sqrt(np.sum(d_centroid ** 2, axis=0)))
    T = np.eye(Ncoords)
    T[0, 0] = s
    T[1, 1] = s
    T[0, 2] = -mean[0] * s
    T[1, 2] = -mean[1] * s
    points_norm = T @ points

    return points_norm, T


def get_A(points1, points2):
    A = np.empty(shape=(2*points1.shape[1], 9))

    points1 = normalise_last_coord(points1)
    points2 = normalise_last_coord(points2)

    a1 = -points2[2,:] * points1
    a2 = points2[1,:] * points1
    a3 = -a1
    a4 = -points2[0,:] * points1

    for pidx in range(points1.shape[1]):
        A[2*pidx,:] = [0,0,0,*a1[:,pidx],*a2[:,pidx]]
        A[2*pidx+1,:] = [*a3[:,pidx],0,0,0,*a4[:,pidx]]
    return A


def DLT_homography(points1, points2):
    """
    Compute the homography matrix H that maps points1 to points2 using the normalized DLT algorithm.

    First, it computes the matrix A of the linear system Ah = 0, where h is the vector of the
    homography matrix H stacked as h = [h11, h12, h13, h21, h22, h23, h31, h32, h33].

    Then, it computes the nullspace of A using the SVD decomposition of A. The last column of V
    contains the solution h of Ah = 0. Finally, it reshapes h to obtain the homography matrix H.

    Args:
        points1: 3x4 matrix of homogeneous coordinates of the points in the first image
        points2: 3x4 matrix of homogeneous coordinates of the points in the second image

    Returns:
        H: 3x3 matrix of the homography
    """

    # Normalize (translation and scaling)
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)

    A = get_A(points1_norm, points2_norm)

    _, _, V = np.linalg.svd(A)
    h = np.transpose(V)[:,-1]
    H = h.reshape([3,3])
    H_norm = np.linalg.inv(T2) @ H @ T1

    return H_norm


def Inliers(H, points1, points2, th) -> np.ndarray:
    """
    Compute the number of inliers consistent with H by the number of 
    correspondences for which d < th pixels.

    Args:
        H: 3x3 matrix of the homography
        points1: 3xN matrix of homogeneous coordinates of the points in the first image
        points2: 3xN matrix of homogeneous coordinates of the points in the second image
        th: threshold for the distance between the points

    Returns:
        idx: indices of the inliers
    """
    # Check that H is invertible
    if abs(math.log(np.linalg.cond(H))) > 15:
        idx = np.empty(1)
        return idx

    points1_projected = H @ points1
    points2_projected = np.linalg.inv(H) @ points2
    points1_normalized = normalise_last_coord(points1)[:2, :]
    points2_normalized = normalise_last_coord(points2)[:2, :]
    points1_projected_normalized = normalise_last_coord(points1_projected)[:2, :]
    points2_projected_normalized = normalise_last_coord(points2_projected)[:2, :]

    d1 = np.linalg.norm(points2_normalized - points1_projected_normalized, axis=0)
    d2 = np.linalg.norm(points2_projected_normalized - points1_normalized, axis=0)
    d = d1 + d2

    idx = np.where(d < th)[0]
    return idx


def Ransac_DLT_homography(points1: np.ndarray, points2: np.ndarray, th: int, max_it: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the homography matrix H that maps points1 to points2 using the RANSAC algorithm.

    Args:
        points1: 3xN matrix of homogeneous coordinates of the points in the first image
        points2: 3xN matrix of homogeneous coordinates of the points in the second image
        th: threshold for the distance between the points (for inliers)
        max_it: maximum number of iterations

    Returns:
        H: 3x3 matrix of the homography
        best_inliers: indices of the inliers
    """
    Ncoords, Npts = points1.shape

    it = 0
    best_inliers = np.empty(1)

    while it < max_it:
        indices = random.sample(range(1, Npts), 4)
        H = DLT_homography(points1[:,indices], points2[:,indices])
        inliers = Inliers(H, points1, points2, th)

        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
 
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 -  fracinliers**4
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)

        it += 1

    # compute H from all the inliers
    H = DLT_homography(points1[:,best_inliers], points2[:,best_inliers])
    inliers = best_inliers

    return H, inliers


def kp_match(img1, img2, descr='orb', dist_th=0.85):

    if descr=='orb':
        descr = cv2.ORB_create()
    else:
        descr = cv2.SIFT_create()   

    kp1, d1 = descr.detectAndCompute(img1, None)
    kp2, d2 = descr.detectAndCompute(img2, None)

    # match keypoints
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(d1, d2, k=2)
    # Filter with Lowe's ratio test
    _matches = []
    for m,n in matches:
        if m.distance < dist_th * n.distance:
            _matches.append([m])

    return kp1, kp2, _matches


def get_inliers_homography(kp1, kp2, matches):
    P1 = []
    P2 = []
    for m in matches:
        P1.append([kp1[m[0].queryIdx].pt[0], kp1[m[0].queryIdx].pt[1], 1])
        P2.append([kp2[m[0].trainIdx].pt[0], kp2[m[0].trainIdx].pt[1], 1])

    P1 = np.asarray(P1)
    P1 = P1.T
    P2 = np.asarray(P2)
    P2 = P2.T

    H_, idxs_inlier_matches = Ransac_DLT_homography(P1, P2, 3, 1000)

    return P1, P2, H_, idxs_inlier_matches


def create_mosaic(I1, I2, I3, corners, H_12, H_23, out_filename='', store=False):
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
    I3 = cv2.cvtColor(I3, cv2.COLOR_BGR2RGB)

    I1_w = apply_H_fixed_image_size(I1, H_12, corners)
    I2_w = apply_H_fixed_image_size(I2, np.eye(3), corners)
    I3_w = apply_H_fixed_image_size(I3, np.linalg.inv(H_23), corners)

    img_mosaic = np.maximum(I3_w,np.maximum(I1_w,I2_w))
    plot_img(img_mosaic)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    img_mosaic = cv2.cvtColor(img_mosaic, cv2.COLOR_RGB2BGR)

    if store:
        cv2.imwrite(out_filename, img_mosaic)
    return


def geometric_error_terms(variables: np.ndarray, data_points: np.ndarray) -> np.ndarray:
    """
    Compute the vector of residuals as the reprojection error (without squaring the terms, as 
    the function 'least squares' constructs the cost function as a sum of squares of the residuals).

    Args:
        variables: 1D array of variables to be optimized. The first 9 elements are the 3x3 homography H.
                    The remaining elements are the 2D coordinates of the keypoints in the first image.
        data_points: tuple of two 3xN matrices of homogeneous coordinates of the points in the first and second image, respectively. 
    """
    points1, points2 = data_points
    # Homography that projects points1 to points2
    H = variables[:9].reshape(3,3)

    # Initial guess of keypoint locations, can be computed with: np.linalg.inv(H) @ points2
    points1_projected = variables[9:].reshape(2,-1)
    points1_h_hom = np.vstack([points1_projected, np.ones(shape=(1,points1_projected.shape[1]))])
    
    # Project points1 to the image 2
    points2_projected = H @ points1_h_hom
    points2_projected = normalise_last_coord(points2_projected)[:2, :]

    # Compute the reprojection error
    err1 = (points1 - points1_projected)
    err2 = (points2 - points2_projected)
    
    return np.concatenate([err1.flatten(), err2.flatten()])


def get_geom_error_inputs(points1, points2, H_12, idxs_inlier_matches) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the inputs of the function 'geometric_error_terms' for the optimization problem.
    """
    # Get inlier keypoints from images 1 and 2
    points1_inl = points1[:,idxs_inlier_matches]
    points2_inl = points2[:,idxs_inlier_matches]

    # Initial guess of the keypoint locations
    points1_proj = np.linalg.inv(H_12) @ points2_inl

    # 1D array of variables to be optimized (the homography H and the keypoint locations)
    variables = np.concatenate([H_12.flatten(), points1_proj[:2,:].flatten()])

    # Join data points for the argument of the function 'geometric_error_terms'
    points = [points1_inl[:2,:], points2_inl[:2,:]]

    return variables, points, points1_proj


def get_result_solution_ls(result: object):
    """
    Get the solution of the optimization problem
    from the result of the function 'least_squares'.
    """
    H = result.x[:9].reshape(3,3)
    points = result.x[9:].reshape(2,-1)
    return H, points


def compute_geometric_error(points: np.ndarray, H: np.ndarray, points1_proj: np.ndarray) -> float:
    """
    Compute the geometric error as the sum of the squared reprojection errors.

    Args:
        points: tuple of two 3xN matrices of homogeneous coordinates of the points in the first and second image, respectively. 
        H: 3x3 homography matrix.
        points1_proj: 2xN matrix of homogeneous coordinates of the points in the first image, projected to the second image.
    """
    variables = np.concatenate([H.flatten(), points1_proj[:2,:].flatten()])
    error = geometric_error_terms(variables, points)
    return np.sum(error**2)


def show_points_refined(img, pointsA, pointsA_ref):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(pointsA[0,:],pointsA[1,:], c='cyan', marker='+')
    ax.scatter(pointsA_ref[0,:],pointsA_ref[1,:], c='fuchsia', marker='+')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()
    return


def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o


def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:,:3]) @ np.array([x[0], x[1], 1])
    return v


def plot_camera(P, w, h, fig, legend):
    
    o = optical_center(P)
    scale = 200
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale

    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0]+p2[0])/2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1]+p2[1])/2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2]+p2[2])/2])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))



def plot_image_origin(w, h, fig, legend):
    p1 = np.array([0, 0, 0])
    p2 = np.array([w, 0, 0])
    p3 = np.array([w, h, 0])
    p4 = np.array([0, h, 0])
    
    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]])
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]])
    z = np.array([p1[2], p2[2], p3[2], p4[2], p1[2]])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
