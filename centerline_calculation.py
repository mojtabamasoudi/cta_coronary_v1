import sys
import os
import numpy as np
import nibabel as nib
from skimage.morphology import ball, binary_dilation
import pandas as pd
from math import *
import time
from cv2 import dilate, erode
from numpy import genfromtxt
from centreline_labelling_tools import *


def main():
    tic = time.time()
    niftidir = r'D:\Mojtaba\Dataset_test\dataset02\005_chen_chuanli_ct1570562'

    aorta_name = 'aorta.nii.gz'
    kernel_size = 2  # dilation for connecting aorta and arteries

    niftiname = 'coronary.nii.gz'

    niftiname_L = 'L_coronary.nii.gz'
    niftiroot_L = 'L_coronary'

    niftiname_R = 'R_coronary.nii.gz'
    niftiroot_R = 'R_coronary'

    # # # # # # # # # data preparation # # # # # # # # #

    # left and right seperation
    filename1 = os.path.join(niftidir, niftiname)
    img1 = nib.load(filename1)
    coronary = img1.get_fdata()
    coronary = coronary.astype(np.float64)

    left_cor = np.where(coronary >= 1.5, 1, 0)

    # 3d dilation
    kernel = ball(kernel_size)
    dilated_left_cor = binary_dilation(left_cor, kernel)

    mask = nib.Nifti1Image(dilated_left_cor, img1.affine, img1.header)
    nib.save(mask, os.path.join(niftidir, niftiname_L))

    temp = np.where(coronary >= 1.5, 0, 1)
    right_cor = coronary * temp

    # 3d dilation
    dilated_right_cor = binary_dilation(right_cor, kernel)

    mask = nib.Nifti1Image(dilated_right_cor, img1.affine, img1.header)
    nib.save(mask, os.path.join(niftidir, niftiname_R))

    # generating intersection mask between aorta and right, left side
    # this intersection can be used for opening surface
    filename2 = os.path.join(niftidir, aorta_name)
    img2 = nib.load(filename2)
    aorta = img2.get_fdata()
    aorta = aorta > 0
    aorta = aorta.astype(np.float64)


    kernel_dilation = 11
    kernel = np.ones((kernel_dilation, kernel_dilation), np.uint8)
    dilated_aorta = dilate(aorta, kernel)

    # Left side

    intersect = dilated_left_cor * dilated_aorta

    slice_index = np.argwhere(intersect == 1)
    two_surface = np.zeros(np.shape(intersect))
    x = slice_index[0, 0]
    y = slice_index[0, 1]
    z = slice_index[0, 2]

    # kernel = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    size_surface = 5

    if z >= size_surface:

        kernel = np.ones([size_surface * 2 + 1, size_surface * 2 + 1, size_surface * 2 + 1])

        two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
        z - size_surface:z + (size_surface+1)] = kernel
        two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
        z - (size_surface-1):z + (size_surface+2)] = kernel

    else:

        kernel = np.ones([size_surface * 2 + 1, size_surface * 2 + 1, size_surface + 1])

        two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
        z :z + (size_surface+1)] = kernel


    mask = nib.Nifti1Image(two_surface, img1.affine, img1.header)

    intersect_L = niftiroot_L +'_intersect.nii.gz'
    nib.save(mask, os.path.join(niftidir, intersect_L))

    # Right side

    intersect = dilated_right_cor * dilated_aorta

    slice_index = np.argwhere(intersect == 1)
    two_surface = np.zeros(np.shape(intersect))
    x = slice_index[0, 0]
    y = slice_index[0, 1]
    z = slice_index[0, 2]

    if z >= size_surface:

        kernel = np.ones([size_surface * 2 + 1, size_surface * 2 + 1, size_surface * 2 + 1])

        two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
        z - size_surface:z + (size_surface+1)] = kernel
        two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
        z - (size_surface-1):z + (size_surface+2)] = kernel

    else:

        kernel = np.ones([size_surface * 2 + 1, size_surface * 2 + 1, size_surface + 1])

        two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
        z :z + (size_surface+1)] = kernel

    mask = nib.Nifti1Image(two_surface, img1.affine, img1.header)

    intersect_R = niftiroot_R +'_intersect.nii.gz'
    nib.save(mask, os.path.join(niftidir, intersect_R))

    # # # # # # # # # branch/sub-branch labeling # # # # # # # # #

    # calculation left side centerline
    csv_file_L = centerline_calculation(niftidir, niftiname_L, niftiroot_L, intersect_L)

    # calculation right side centerline
    csv_file_R = centerline_calculation(niftidir, niftiname_R, niftiroot_R, intersect_R)

    # calculation of endpoints, co_occurrence matrix and labels near to aorta
    # left side
    co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal, aorta_z = endpoint_calculation\
        (niftidir, np.shape(coronary), csv_file_L, niftiroot_L, intersect_L)

    # labeling branches
    # left side
    branch_labeling_left(niftidir, csv_file_L, co_occurrence, co_occurrence_inverse,
                                  labels_near_aorta, label_proximal, aorta_z)

    # calculation of endpoints, co_occurrence matrix and labels near to aorta
    # right sideq
    co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal, aorta_z = endpoint_calculation\
        (niftidir, np.shape(coronary), csv_file_R, niftiroot_R, intersect_R)

    # labeling branches
    # right side
    branch_labeling_right(niftidir, csv_file_R, co_occurrence, co_occurrence_inverse,
                                  labels_near_aorta, label_proximal, aorta_z)

    # # # # # # # # # CMPR preparation # # # # # # # # #
    # right
    dilated_right_cor_aorta = dilated_right_cor + dilated_aorta
    dilated_right_cor_aorta = dilated_right_cor_aorta > 0

    mask = nib.Nifti1Image(dilated_right_cor_aorta, img1.affine, img1.header)

    dilated_right_cor_aorta_name = 'dilated_right_cor_aorta.nii.gz'
    nib.save(mask, os.path.join(niftidir, dilated_right_cor_aorta_name))

    # left
    dilated_left_cor_aorta = dilated_left_cor + dilated_aorta
    dilated_left_cor_aorta = dilated_left_cor_aorta > 0

    mask = nib.Nifti1Image(dilated_left_cor_aorta, img1.affine, img1.header)

    dilated_left_cor_aorta_name = 'dilated_left_cor_aorta.nii.gz'
    nib.save(mask, os.path.join(niftidir, dilated_left_cor_aorta_name))

    # center of aorta
    import cv2
    a = np.argwhere(aorta == 1)
    minimum = np.min(a[:, 2])
    maximum = np.max(a[:, 2])
    cz = int(np.round((maximum + minimum) / 2))
    mid_slice = aorta[:, :, cz]
    M = cv2.moments(mid_slice)
    cy = int(M["m10"] / M["m00"])
    cx = int(M["m01"] / M["m00"])

    csv_file_L = 'centreline_' + niftiroot_L + '.csv'
    csv_endpoint_file = 'endpoint_' + csv_file_L
    cmpr_preparation(niftidir, niftiroot_L, dilated_left_cor_aorta_name, csv_endpoint_file, cx, cy, cz)

    csv_file_R = 'centreline_' + niftiroot_R + '.csv'
    csv_endpoint_file = 'endpoint_' + csv_file_R
    cmpr_preparation(niftidir, niftiroot_R, dilated_right_cor_aorta_name, csv_endpoint_file, cx, cy, cz)

    toc = time.time()
    print((toc - tic) / 60)

if __name__ == '__main__':
    main()


