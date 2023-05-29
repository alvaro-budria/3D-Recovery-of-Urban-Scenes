import numpy as np
import random 
import sys
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA


def optical_center(P):
    u, s, vh = np.linalg.svd(P)
    o = vh[:, -1]
    o = o[:3] / o[3]
    return o


def view_direction(P, x):
    v, resid, rank, s = np.linalg.lstsq(P[:, :3], x, rcond=None)
    return v


def draw_lines(points, ax: Optional[Axes3D] = None, **line_kwargs):
    """
    :param points: shape (num_points, 2, 3)
    :param ax:
    :return:
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
    assert isinstance(ax, Axes3D)

    for l in points:
        p1, p2 = l[0], l[1]
        ax.plot(xs=[p1[0], p2[0]], ys=[p1[1], p2[1]], zs=[p1[2], p2[2]], **line_kwargs)


def get_sight_line(P, w, h, scale):
    o = optical_center(P)
    p1 = o + view_direction(P, np.array([0, 0, 1])) * scale
    p2 = o + view_direction(P, np.array([w, 0, 1])) * scale
    p3 = o + view_direction(P, np.array([w, h, 1])) * scale
    p4 = o + view_direction(P, np.array([0, h, 1])) * scale
    points = np.array([p1, p2, p3, p4])
    n = points.mean(axis=0) - o
    n = n / np.linalg.norm(n)
    return o, n


def get_camera_frame_points(P, w, h, scale, ax=None):
    """
    :param P: Camera matrix
    :param w: Width
    :param h: Height
    :param scale: Scale
    :return:
    """
    o = optical_center(P)
    p1 = o + view_direction(P, np.array([0, 0, 1])) * scale
    p2 = o + view_direction(P, np.array([w, 0, 1])) * scale
    p3 = o + view_direction(P, np.array([w, h, 1])) * scale
    p4 = o + view_direction(P, np.array([0, h, 1])) * scale

    points = np.array([[o, p1]])
    points = np.vstack((points, np.array([[o, p2]])))
    points = np.vstack((points, np.array([[o, p3]])))
    points = np.vstack((points, np.array([[o, p4]])))
    points = np.vstack((points, np.array([[p1, p2]])))
    points = np.vstack((points, np.array([[p2, p3]])))
    points = np.vstack((points, np.array([[p3, p4]])))
    points = np.vstack((points, np.array([[p4, p1]])))
    return points


def plot_camera(P, w, h, scale, ax=None, **plot_kwargs):
    """
    :param P: Camera matrix
    :param w: Width
    :param h: Height
    :param scale: Scale
    :param ax: matplotlib axis
    :return:
    """
    points = get_camera_frame_points(P, w, h, scale)
    draw_lines(points, ax=ax, **plot_kwargs)


def projective2img(x):
    """
    Take a 3D homogenous/projective coordinate and obtain the 2D (image) euclidean equivalent
    """
    assert x.shape[0] == 3, f'`x` shape {x.shape} expected (3,...)'
    return x[:2, ...] / x[2][np.newaxis]


def homogeneous2euclidean(x):
    """
    Take a 4D homogenous coordinate and normalize to obtain the 3D euclidan equivalent
    Take a 3D homogenous/projective coordinate and obtain the 2D (image) euclidean equivalent
    """
    assert x.shape[0] == 4, f'`x` shape {x.shape} expected (4,...)'
    return x[:3, ...] / x[3][np.newaxis]


def draw_points(points, ax=None, **plot_kwargs):
    # Creating figure
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], **plot_kwargs)


def normalize(x, imsize):
    H = np.array(
        [
            [2 / imsize[1], 0, -1],
            [0, 2 / imsize[0], -1],
            [0, 0, 1],
        ]
    )
    return H @ x


def setup_equations(x1, x2, P1, P2):
    return np.array(
        [
            x1[0] * P1[2] - P1[0],
            x1[1] * P1[2] - P1[1],
            x2[0] * P2[2] - P2[0],
            x2[1] * P2[2] - P2[1],
        ]
    )


def compute_DLT(x1, x2, P1, P2):
    X = []

    for p1, p2 in zip(x1.T, x2.T):
        A = setup_equations(p1, p2, P1, P2)

        _, _, Vh = np.linalg.svd(A)

        X.append(Vh[-1, :])

    X = np.vstack(X)
    return X.T


# Write here the method for DLT triangulation
def triangulate(x1, x2, P1, P2, imsize) -> np.ndarray:
    assert P1.shape == (3,4) == P2.shape
    assert x1.shape == x2.shape and x1.shape[0] == 3

    # Normalize points in both images
    x1, x2 = x1 / x1[2, :], x2 / x2[2, :]
    x1, x2, P1, P2 = normalize(x1, imsize), normalize(x2, imsize), normalize(P1, imsize), normalize(P2, imsize)
    X = compute_DLT(x1, x2, P1, P2)
    return X


def normalize_last_coord(x):
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

    P1_norm = normalize_last_coord(P1_norm)
    P2_norm = normalize_last_coord(P2_norm)

    Ncoords, Npts = P1_norm.shape
    W = np.zeros((Npts, 9))

    for i in range(Npts):
        u1, v1, _ = P1_norm[:, i]
        u2, v2, _ = P2_norm[:, i]
        W[i, :] = [u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, 1]

    U, S, Vt = LA.svd(W)
    f = Vt[-1, :]
    F = f.reshape((3, 3))
    F = F / F[2, 2]

    # Enforce constraint that fundamental matrix has rank 2
    U, d, Vt = np.linalg.svd(F)
    D = np.zeros((3,3))
    D[0,0] = d[0]
    D[1,1] = d[1]
    F = U @ D @ Vt

    # Denormalise
    F = T2.T @ F @ T1
    return F


def compute_inliers(F, x1, x2, th):
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
    Fx1 = F @ x1
    Ftx2 = F.T @ x2

    n = x1.shape[1]
    x2tFx1 = np.zeros((1, n))

    for i in range(n):
        x2t = x2[:,i]
        x2t = x2t.T
        x2tFx1[0,i] = x2t @ F @ x1[:,i]

    # evaluate distances
    den = Fx1[0,:]**2 + Fx1[1,:]**2 + Ftx2[0,:]**2 + Ftx2[1,:]**2
    den = den.reshape((1, n))

    d = x2tFx1**2 / den
    inliers_indices = np.where(d[0,:] < th)
    return inliers_indices[0]


def ransac_fundamental_matrix(P1, P2, th, max_it_0):
    """
    RANSAC algorithm to estimate the fundamental matrix
    """
    Ncoords, Npts = P1.shape

    it = 0
    max_it = max_it_0
    best_inliers = np.empty(1)

    while it < max_it:
        indices = random.sample(range(1, Npts), 8)
        F = fundamental_matrix(P1[:,indices], P2[:,indices])
        inliers = compute_inliers(F, P1, P2, th)

        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
 
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 - fracinliers**8
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = np.log(1-p)/np.log(pNoOutliers)

        it += 1
        max_it_0 = min(max_it, max_it_0)

    # compute H from all the inliers
    F = fundamental_matrix(P1[:, best_inliers], P2[:, best_inliers])

    return F, best_inliers


def essential_from_fundamental(K, F):
    return K.T @ F @ K


def reflect_rotation(R):
    print("_"*25)
    if np.linalg.det(R) < 0:
        print("Reflecting rotation matrix!")
        return -R
    return R