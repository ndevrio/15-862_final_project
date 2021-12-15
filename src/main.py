import numpy as np
import os
import cv2
from skimage import io, color
from matplotlib import pyplot as plt
from scipy import signal
from scipy import interpolate, ndimage
from skimage.color import rgb2gray
from skimage.color import gray2rgb
import os
import glob
from cp_hw6 import *



num_pics = 46
proj_r = 1080
proj_c = 1920
ds = 10


cam_img = np.zeros((4000, 6000))
proj_img = np.zeros((1080, 1920))

# Matrices obtained from calibration.py
ip_cam = np.array([[5.91174608e+03, 0.00000000e+00, 3.02680976e+03],
                   [0.00000000e+00, 5.90339820e+03, 2.14826772e+03],
                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_cam = np.array([[-1.74192015e-01, 4.50120113e+00, 9.80591234e-03, -3.17271527e-03, -4.62783728e+01]])
ip_proj = np.array([[2.77659196e+03, 0.00000000e+00, 8.57121942e+02],
                    [0.00000000e+00, 2.75408401e+03, 1.25669564e+03],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_proj = np.array([[-0.00762913, -0.93583331, -0.00960655, -0.00687727, 3.18292129]])

rmat = np.array([[0.98119266, -0.04345108, 0.18807701],
                 [0.06693766, 0.99046785, -0.12038599],
                 [-0.18105333, 0.13071129, 0.9747483 ]])
tvec = np.array([[-115.37108278], [-109.7766752], [128.38810639]])


cam_image_corners = np.float32([[0,0], [cam_img.shape[1], 0], [0, cam_img.shape[0]], [cam_img.shape[1], cam_img.shape[0]]])
cam_corner_colors = [(1,0,0), (1,0,0), (1,0,0), (1,0,0)]
cam_corner_rays = np.squeeze(1000*pixel2ray(np.float32(cam_image_corners), ip_cam, dist_cam)).T

proj_image_corners = np.float32([[0,0], [proj_img.shape[1], 0], [0, proj_img.shape[0]], [proj_img.shape[1], proj_img.shape[0]]])
proj_corner_colors = [(0,0,1), (0,0,1), (0,0,1), (0,0,1)]
proj_corner_rays = np.matmul(rmat.T, np.squeeze(1000*pixel2ray(np.float32(proj_image_corners), ip_proj, dist_proj)).T)


def flip(c):
    return '1' if(c == '0') else '0'


def graytoBinary(gray):
 
    binary = ""
 
    # MSB of binary code is same
    # as gray code
    binary += gray[0]
 
    # Compute remaining bits
    for i in range(1, len(gray)):
         
        # If current bit is 0,
        # concatenate previous bit
        if (gray[i] == '0'):
            binary += binary[i - 1]
 
        # Else, concatenate invert
        # of previous bit
        else:
            binary += flip(binary[i - 1])
 
    return binary


def stacked_lstsq(L, b, rcond=1e-10):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond*s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1/s[s>=s_min]
    x = np.einsum('...ji,...j->...i', v,
                  inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
    #return np.diagonal(np.conj(x, x), axis1=1, axis2=2)
    return np.conj(x, x)


def nearest_intersection(points, dirs):
    """
    :param points: (N, 2, 3) array of points on the lines
    :param dirs: (N, 2, 3) array of unit direction vectors
    :returns: (N, 3,) array of intersection point
    """
    dirs = dirs / np.expand_dims(np.linalg.norm(dirs, axis=2), 2)
    dirs_mat = dirs[:, :, :, np.newaxis] @ dirs[:, :, np.newaxis, :]
    points_mat = points[:, :, :, np.newaxis]
    I = np.eye(3)

    #print((I - dirs_mat).sum(axis=1).shape)
    #print(((I - dirs_mat) @ points_mat).sum(axis=1)[..., 0].shape)
    return stacked_lstsq(
        (I - dirs_mat).sum(axis=1),
        ((I - dirs_mat) @ points_mat).sum(axis=1)[..., 0]
    )


def main(I, I_color):
    col_decode = [44, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    #row_decode.reverse()
    row_decode = [44, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
    #col_decode.reverse()

    I_ave = np.mean(I, axis=0)
    non_illum_mask = ((I[44]-I_ave) < 0.07)
    # Display
    #plt.imshow(non_illum_mask)
    #plt.show()

    decoded_row_str = np.chararray(I[0].shape, unicode=True)
    decoded_row_bits = np.zeros(I[0].shape)
    decoded_col_str = np.chararray(I[0].shape, unicode=True)
    decoded_col_bits = np.zeros(I[0].shape)

    n_r_bins = 32
    r_bins = np.floor_divide(np.arange(proj_r), proj_r/n_r_bins)
    r_bins = (r_bins / n_r_bins) * proj_r

    n_c_bins = 48
    c_bins = np.floor_divide(np.arange(proj_c), proj_c/n_c_bins)
    c_bins = (c_bins / n_c_bins) * proj_c

    for i in range(len(row_decode)):
        select = (I[row_decode[i]] < I[row_decode[i]+1])
        high_bits = np.chararray(I[0].shape, unicode=True)
        high_bits[:] = '1'
        high_bits[np.logical_or(select, non_illum_mask)] = '0'

        # Display
        """plt.subplot(121)
        plt.imshow(I[row_decode[i]+1], cmap='gray')
        plt.subplot(122)
        plt.imshow(high_bits, cmap='gray')
        plt.show()"""

        decoded_row_str = np.char.add(decoded_row_str, high_bits)
        #decoded_row_bits += high_bits * int(2**i)

    # Decode from gray code
    for r in range(decoded_row_str.shape[0]):
        for c in range(decoded_row_str.shape[1]):
            decoded_row_str[r, c] = graytoBinary(decoded_row_str[r, c])
            decoded_row_bits[r, c] = int(decoded_row_str[r, c], 2)

    # Convert from binary to projector row/column
    decoded_row_bits[decoded_row_bits == 0] = 4095
    decoded_row_bits = np.rint(proj_r - ((decoded_row_bits / 4096) * proj_r)).astype(int) # r_bins[]
    print('Decoded rows')

    for i in range(len(col_decode)):
        select = (I[col_decode[i]] < I[col_decode[i]+1])
        high_bits = np.chararray(I[0].shape, unicode=True)
        high_bits[:] = '1'
        high_bits[np.logical_or(select, non_illum_mask)] = '0'

        # Display
        """plt.subplot(121)
        plt.imshow(I[col_decode[i]+1], cmap='gray')
        plt.subplot(122)
        plt.imshow(high_bits, cmap='gray')
        plt.show()"""

        decoded_col_str = np.char.add(decoded_col_str, high_bits)
        #decoded_col_bits += high_bits * int(2**i)

    # Decode from gray code
    for r in range(decoded_col_str.shape[0]):
        for c in range(decoded_col_str.shape[1]):
            decoded_col_str[r, c] = graytoBinary(decoded_col_str[r, c])
            decoded_col_bits[r, c] = int(decoded_col_str[r, c], 2)

    # Convert from binary to projector row/column
    decoded_col_bits[decoded_col_bits == 0] = 4095
    decoded_col_bits = np.rint(proj_c - ((decoded_col_bits / 4096) * proj_c)).astype(int) # c_bins[]
    print('Decoded columns')

    # Display
    """plt.subplot(121)
    plt.imshow(decoded_row_bits, cmap='jet')
    plt.subplot(122)
    plt.imshow(decoded_col_bits, cmap='jet')
    plt.show()"""

    print('Calculating projections')
    #proj_origin = -tvec
    proj_origin = np.matmul(-np.linalg.inv(rmat), tvec)

    decoded_proj_rc = np.stack([decoded_col_bits, decoded_row_bits])
    #decoded_proj_rc = np.moveaxis(decoded_proj_rc, 1, 2)  ### SWITCH ###
    decoded_proj_rc = np.moveaxis(decoded_proj_rc, 0, 2)
    print(decoded_proj_rc.shape)
    print(np.max(decoded_proj_rc), np.min(decoded_proj_rc))
    decoded_proj_rc = np.reshape(decoded_proj_rc, (decoded_proj_rc.shape[0]*decoded_proj_rc.shape[1], 2))
    #proj_rays = np.matmul(rmat.T, np.squeeze(1*pixel2ray(np.float32(decoded_proj_rc), ip_proj, dist_proj)).T)  # [N, 2]
    proj_rays = np.squeeze(1*pixel2ray(np.float32(decoded_proj_rc), ip_proj, dist_proj)).T  # [N, 2]
    #proj_rays = np.squeeze(400*pixel2ray(np.float32(decoded_proj_rc), ip_proj, dist_proj)).T  # [N, 2]
    proj_rays = np.linalg.inv(rmat) @ (proj_rays - tvec)
    #proj_rays = (rmat @ proj_rays) - tvec

    proj_rays = proj_rays - proj_origin

    cam_idx = np.indices((cam_img.shape[1], cam_img.shape[0]))[:, ::ds, ::ds]
    #cam_idx = np.moveaxis(cam_idx, 1, 2)   ### SWITCH ###
    cam_idx = np.moveaxis(cam_idx, 0, 2)
    cam_idx = np.reshape(cam_idx, (cam_idx.shape[0]*cam_idx.shape[1], 2))
    print(cam_idx.shape)
    print(cam_idx[:10])
    print(cam_idx[-10:])
    cam_rays = np.squeeze(pixel2ray(np.float32(cam_idx), ip_cam, dist_cam)).T  # [N, 2]


    proj_points = np.zeros((proj_rays.shape)).T
    proj_points[:] = proj_origin[:, 0]
    cam_points = np.zeros((cam_rays.shape)).T


    points = np.stack([proj_points, cam_points])
    points = np.moveaxis(points, 0, 1)
    dirs = np.stack([proj_rays.T, cam_rays.T])
    dirs = np.moveaxis(dirs, 0, 1)

    print('Performing reconstruction')
    recon_points = nearest_intersection(points, dirs)

    recon_points[np.linalg.norm(recon_points, axis=1) > 1000] = 0
    non_illum_mask = np.reshape(non_illum_mask, (non_illum_mask.shape[0]*non_illum_mask.shape[1]))
    recon_points[non_illum_mask] = 0

    # Display
    ds_plot = 2
    fig = plt.figure("Reconstructed point cloud")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ###
    print(cam_img.shape[1])
    #cam_image_corners = np.float32([[0,0], [cam_img.shape[0], 0], [0, cam_img.shape[1]], [cam_img.shape[0], cam_img.shape[1]]])
    cam_image_corners = np.float32([[0,0], [cam_img.shape[1], 0], [0, cam_img.shape[0]], [cam_img.shape[1], cam_img.shape[0]]])
    cam_corner_colors = [(1,0,0), (1,0,0), (1,0,0), (1,0,0)]
    cam_corner_rays = np.squeeze(1000*pixel2ray(np.float32(cam_image_corners), ip_cam, dist_cam)).T

    #proj_image_corners = np.float32([[0,0], [proj_img.shape[0], 0], [0, proj_img.shape[1]], [proj_img.shape[0], proj_img.shape[1]]])
    proj_image_corners = np.float32([[0,0], [proj_img.shape[1], 0], [0, proj_img.shape[0]], [proj_img.shape[1], proj_img.shape[0]]])
    proj_corner_colors = [(0,0,1), (0,0,1), (0,0,1), (0,0,1)]
    #proj_corner_rays = np.matmul(rmat.T, np.squeeze(1000*pixel2ray(np.float32(proj_image_corners), ip_proj, dist_proj)).T)
    proj_corner_rays = np.squeeze(1000*pixel2ray(np.float32(proj_image_corners), ip_proj, dist_proj)).T
    proj_corner_rays = np.linalg.inv(rmat) @ (proj_corner_rays - tvec)
    #proj_corner_rays = rmat @ (proj_corner_rays - tvec)
    ###

    scatter_colors = np.reshape(I_color[44], (I_color[44].shape[0]*I_color[44].shape[1], 3))
    scatter_colors = scatter_colors[np.linalg.norm(recon_points, axis=1) != 0]
    recon_points = recon_points[np.linalg.norm(recon_points, axis=1) != 0]

    #ax.scatter(0, 0, 0, s=10, marker="s", color=(1,0,0))
    #ax.scatter(proj_origin[0], proj_origin[1], proj_origin[2], s=10, marker="s", color=(0,0,1))
    ax.scatter(recon_points[::ds_plot, 0], recon_points[::ds_plot, 1], recon_points[::ds_plot, 2], s=2, marker="o", color=scatter_colors[::ds_plot])
    
    ray_colors = [(0,1,1)]*4
    p1 = 64330
    p2 = 64334
    ax.quiver(0, 0, 0, cam_corner_rays[0,:], cam_corner_rays[1,:], cam_corner_rays[2,:], color=cam_corner_colors, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 1000*cam_rays[0,-4], 1000*cam_rays[1,-4], 1000*cam_rays[2,-4], color=ray_colors, arrow_length_ratio=0.05)
    ax.quiver(proj_origin[0], proj_origin[1], proj_origin[2], 1000*proj_rays[0,p1:p2], 1000*proj_rays[1,p1:p2], 1000*proj_rays[2,p1:p2], color=[(0,1,0)]*4, arrow_length_ratio=0.05)
    ax.quiver(proj_origin[0], proj_origin[1], proj_origin[2], proj_corner_rays[0,:], proj_corner_rays[1,:], proj_corner_rays[2,:], color=proj_corner_colors, arrow_length_ratio=0.05)


    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()



if __name__ == '__main__':
    # Intersection testing
    """points = np.array([[-1, 0, -1], [1, -2, 1]])
    dirs = np.array([[1, 2, 0], [-1, 2, 0]])

    points = np.stack([points, points, points, points])
    dirs = np.stack([dirs, dirs, dirs, dirs])

    print(points.shape)
    print(dirs.shape)
    print('---')
    inter = nearest_intersection(points, dirs)
    print(inter)

    exit()"""

    ### Initials
    I_gray = np.zeros((num_pics, int(4000/ds), int(6000/ds)))
    I = np.zeros((num_pics, int(4000/ds), int(6000/ds), 3))
    if(not os.path.exists('shoe.npy')):
        for i in range(num_pics):
            id = "graycode_" + str(i)
            print(id)
            I[i] = io.imread(("../data/shoe/" + id + ".jpg"))[::ds, ::ds]
            I_gray[i] = rgb2gray(I[i])
        np.save('shoe.npy', I)
    else:
        I = np.load('shoe.npy')
        for i in range(num_pics):
            I_gray[i] = rgb2gray(I[i])

    main(I_gray / 255.0, I / 255.0)