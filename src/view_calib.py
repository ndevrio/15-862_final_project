import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cp_hw6 import *


cam_img = np.zeros((4000, 6000))
proj_img = np.zeros((1080, 1920))

ip_cam = np.array([[5.91174608e+03, 0.00000000e+00, 3.02680976e+03],
                   [0.00000000e+00, 5.90339820e+03, 2.14826772e+03],
                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],])
dist_cam = np.array([[-1.74192015e-01, 4.50120113e+00, 9.80591234e-03, -3.17271527e-03, -4.62783728e+01]])
ip_proj = np.array([[2.77659196e+03, 0.00000000e+00, 8.57121942e+02],
                    [0.00000000e+00, 2.75408401e+03, 1.25669564e+03],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_proj = np.array([[-0.00762913, -0.93583331, -0.00960655, -0.00687727, 3.18292129]])

rmat = np.array([[0.98119266, -0.04345108, 0.18807701], # from camera to projector
                 [0.06693766, 0.99046785, -0.12038599],
                 [-0.18105333, 0.13071129, 0.9747483 ]])
tvec = np.array([[-115.37108278], [-109.7766752], [128.38810639]])


cam_image_corners = np.float32([[0,0], [cam_img.shape[1], 0], [0, cam_img.shape[0]], [cam_img.shape[1], cam_img.shape[0]]])
cam_corner_colors = [(1,0,0), (1,0,0), (1,0,0), (1,0,0)]
cam_corner_rays = np.squeeze(1000*pixel2ray(np.float32(cam_image_corners), ip_cam, dist_cam)).T

proj_image_corners = np.float32([[0,0], [proj_img.shape[1], 0], [0, proj_img.shape[0]], [proj_img.shape[1], proj_img.shape[0]]])
proj_corner_colors = [(0,0,1), (0,0,1), (0,0,1), (0,0,1)]
#proj_corner_rays = np.matmul(rmat.T, np.squeeze(1000*pixel2ray(np.float32(proj_image_corners), ip_proj, dist_proj)).T)
proj_corner_rays = np.squeeze(1000*pixel2ray(np.float32(proj_image_corners), ip_proj, dist_proj)).T
proj_corner_rays = (np.linalg.inv(rmat) @ proj_corner_rays) - tvec
#proj_corner_rays = rmat.T @ proj_corner_rays + tvec
#proj_corner_rays = rmat.T @ (proj_corner_rays - tvec)


#visualize camera relative to calibration plane
fig = plt.figure("Projected camera view")
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(0, 0, 0, s=10, marker="s")
ax.quiver(0, 0, 0, cam_corner_rays[0,:], cam_corner_rays[1,:], cam_corner_rays[2,:], color=cam_corner_colors, arrow_length_ratio=0.05)
#C = np.matmul(-rmat.T, tvec)
C = -tvec
ax.scatter(C[0], C[1], C[2], s=10, marker="s")
ax.quiver(C[0], C[1], C[2], proj_corner_rays[0,:], proj_corner_rays[1,:], proj_corner_rays[2,:], color=proj_corner_colors, arrow_length_ratio=0.05)

ax.set_box_aspect([1,1,1])
set_axes_equal(ax)
plt.show()