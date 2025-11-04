# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Transformation utils
"""

import numpy as np

def _normalize_pose(pose):
    """
    Accept pose as:
      - [x,y,z,roll,yaw,pitch] (6)
      - [x,y,z,yaw]            (4) -> roll=0, pitch=0
      - [x,y,yaw]              (3) -> z=0, roll=0, pitch=0
    Returns (x, y, z, roll, yaw, pitch) as floats.
    """
    arr = np.asarray(pose, dtype=float).reshape(-1)
    if arr.size == 6:
        x, y, z, roll, yaw, pitch = arr
    elif arr.size == 4:
        x, y, z, yaw = arr
        roll = 0.0
        pitch = 0.0
    elif arr.size == 3:
        x, y, yaw = arr
        z = 0.0
        roll = 0.0
        pitch = 0.0
    else:
        raise ValueError(f"Unexpected pose length {arr.size}; expected 3, 4, or 6.")
    return float(x), float(y), float(z), float(roll), float(yaw), float(pitch)

def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to CARLA world system.

    Accepts:
      - pose vectors of length 6/4/3 (x,y,z,roll,yaw,pitch / x,y,z,yaw / x,y,yaw)
      - flattened matrices of length 16 (4x4) or 12 (3x4)
    """
    arr = np.asarray(pose, dtype=float).reshape(-1)

    # Already a matrix?
    if arr.size == 16:
        return arr.reshape(4, 4)
    if arr.size == 12:
        M = arr.reshape(3, 4)
        M = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])
        return M

    # Otherwise: treat as a pose vector
    x, y, z, roll, yaw, pitch = _normalize_pose(arr)

    # Rotation terms
    c_y = np.cos(np.radians(yaw)); s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll)); s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch)); s_p = np.sin(np.radians(pitch))

    # SE(3) matrix
    matrix = np.identity(4)
    matrix[0, 3], matrix[1, 3], matrix[2, 3] = x, y, z

    # R = Rz(yaw) * Rx(roll) * Ry(pitch) (as used by OpenCOOD)
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    
    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous
