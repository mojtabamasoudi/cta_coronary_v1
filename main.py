# IMPORTS --------------------------------------------------------------------------------------------------------------

import sys
import os
from vmtk import pypes
from vmtk import vmtkscripts
import numpy as np
import nibabel as nib
import pandas as pd
import math
from scipy.ndimage import convolve
from skimage.morphology import ball, binary_dilation, max_tree_local_maxima
from coreutils import get_imagepath, get_dirname, get_filename, get_fileroot
# import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# PREAMBLE
# ----------------------------------------------------------------------------------------------------------------------

# # niftipath = os.path.join(os.getcwd(), '3_TOF_3D_multi-slab.nii.gz')
#
# if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
#     niftipath = sys.argv[1]
#
# if 'niftipath' not in locals():
#     print('Select image file...')
#     niftipath = get_imagepath(dialogue_title='Select image file (NIFTI[.gz] or first DICOM in series)')
#
# centreline_visualisation_only = False
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# niftidir = get_dirname(niftipath)
# niftiname = get_filename(niftipath)
# niftiroot = get_fileroot(niftipath)
#
# cwd = os.getcwd()
# os.chdir(niftidir)

from math import *

def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    return a

def getAngleBetweenPoints(x_orig, y_orig, x_landmark, y_landmark):
    deltaY = y_landmark - y_orig
    deltaX = x_landmark - x_orig
    return angle_trunc(atan2(deltaY, deltaX))

niftidir = r'D:\Mojtaba\Dataset_test\9'
os.chdir(niftidir)
niftiname = 'R_coronary.nii.gz'
niftiroot = 'R_coronary'

initial_surface = 'R_coronary_intersect.nii.gz'

# # automatically centerline calculation
# myargs = 'vmtkmarchingcubes -ifile ' + niftiname + ' -l 1.0 -ofile vessel_aorta_surface.vtp ' \
#           '--pipe vmtksurfaceviewer -array GroupIds'
# mypype = pypes.PypeRun(myargs)
#
#
# myargs = 'vmtksurfacesmoothing -iterations 2500 -ifile vessel_aorta_surface.vtp -ofile vessel_aorta_surface_smooth.vtp'
# mypype = pypes.PypeRun(myargs)
#
#
# myargs = 'vmtkmarchingcubes -ifile ' + initial_surface + ' -l 1.0 -ofile aorta_L_intersect_surface.vtp'
# mypype = pypes.PypeRun(myargs)
#
#
# # myargs = 'vmtksurfaceclipper -ifile initial_surface.vtp -ofile OutputFile1.vtp --pipe vmtksurfaceclipper' \
# #          ' -transform @.otransform -ifile vessel_aorta_surface.vtp -ofile OutputFile2.vtp'
# # mypype = pypes.PypeRun(myargs)
#
# myargs = 'vmtksurfacecliploop -ifile vessel_aorta_surface_smooth.vtp' \
#          ' -i2file aorta_L_intersect_surface.vtp -ofile vessel_aorta_surface_clip.vtp'
# mypype = pypes.PypeRun(myargs)
#
#
# myargs = 'vmtksurfaceviewer -ifile vessel_aorta_surface_clip.vtp '
# mypype = pypes.PypeRun(myargs)
#
# myargs = 'vmtksurfaceviewer -ifile vessel_aorta_surface_clip.vtp '
# mypype = pypes.PypeRun(myargs)
#
#
# myargs = 'vmtknetworkextraction -ifile vessel_aorta_surface_clip.vtp -ofile vessel_aorta_centerline.vtp '
#           # '-ographfile output_graph.vtp '
# mypype = pypes.PypeRun(myargs)
#
# myargs = 'vmtklineresampling -ifile vessel_aorta_centerline.vtp -ofile vessel_aorta_centerline1.vtp -length 0.1'
# mypype = pypes.PypeRun(myargs)
#
#
# # myargs = 'vmtksurfaceviewer -ifile output_graph.vtp '
# # mypype = pypes.PypeRun(myargs)
#
# myargs = 'vmtksurfaceviewer -ifile vessel_aorta_centerline1.vtp '
# mypype = pypes.PypeRun(myargs)
#
# myargs = 'vmtksurfacereader -ifile vessel_aorta_surface_clip.vtp --pipe ' + \
#          'vmtkrenderer --pipe ' + \
#          'vmtksurfaceviewer -opacity 0.25' + ' --pipe ' + \
#          'vmtksurfaceviewer -ifile vessel_aorta_centerline1.vtp -array MaximumInscribedSphereRadius'
#
# mypype = pypes.PypeRun(myargs)
#
#
# # ----------------------------------------------------------------------------------------------------------------------
# # EXPORT CENTRELINE COORDINATES
# # ----------------------------------------------------------------------------------------------------------------------
# centreline_file = 'vessel_aorta_centerline1.vtp'
#
#
# centerlineReader = vmtkscripts.vmtkSurfaceReader()
# centerlineReader.InputFileName = centreline_file
# centerlineReader.Execute()
#
# clNumpyAdaptor = vmtkscripts.vmtkCenterlinesToNumpy()
# clNumpyAdaptor.Centerlines = centerlineReader.Surface
# clNumpyAdaptor.Execute()
#
# numpyCenterlines = clNumpyAdaptor.ArrayDict
# points = numpyCenterlines['Points']
#
# labels = numpyCenterlines['CellData']
# cell_points = labels['CellPointIds']
# topology = labels['Topology']
#
# for i in range(np.shape(cell_points)[0]):
#     number = np.shape(cell_points[i])[0]
#     index = np.ones(number) * (i+1)
#     if i == 0:
#         topology_labels = index
#     else:
#         topology_labels = np.append(topology_labels, index, axis=0)
#
#
# myargs = 'vmtkimagereader -ifile  ' + niftiname
# mypype = pypes.PypeRun(myargs)
# RasToIjkMatrixCoefficients = mypype.GetScriptObject('vmtkimagereader', '0').RasToIjkMatrixCoefficients
# XyzToRasMatrixCoefficients = mypype.GetScriptObject('vmtkimagereader', '0').XyzToRasMatrixCoefficients
# ras2ijk = np.asarray(RasToIjkMatrixCoefficients)
# ras2ijk = np.reshape(ras2ijk, (4, 4))
# xyz2ras = np.asarray(XyzToRasMatrixCoefficients)
# xyz2ras = np.reshape(xyz2ras, (4, 4))
#
# for x in range(points.shape[0]):
#     xyz = points[x]
#     xyz = np.append(xyz, 1)
#     ras = np.dot(xyz2ras, xyz)
#     ijk = np.dot(ras2ijk, ras)
#     ijk[3] = topology_labels[x]
#     if x == 0:
#         points_ijk = ijk
#     else:
#         points_ijk = np.append(points_ijk, ijk, axis=0)
#
# points_ijk = np.reshape(points_ijk, [x+1, 4])
# points_ijk = np.round(points_ijk[:, 0:4])
# print(points_ijk)

csv_file = 'centreline_' + niftiroot + '.csv'
# np.savetxt(csv_file, points_ijk, delimiter=",")




# centerline
filename1 = os.path.join(niftidir, niftiname)
img1 = nib.load(filename1)
data1 = img1.get_fdata()

filename2 = os.path.join(niftidir, csv_file)
df = pd.read_csv(filename2)
print(df)

centerline = np.zeros(np.shape(data1))
df_numpy = np.round(df.to_numpy().astype(np.int32))

# TODO
# df_numpy = remove_repetitive(df_numpy)

for i in range(df.values.shape[0]):
    x = df_numpy[i, 0]
    y = df_numpy[i, 1]
    z = df_numpy[i, 2]
    label = df_numpy[i, 3]
    centerline[x, y, z] = label

final = centerline

centerline_name = 'centreline_' + niftiroot + '.nii.gz'
mask = nib.Nifti1Image(final, img1.affine, img1.header)
nib.save(mask, os.path.join(niftidir, centerline_name))


# endpoint candidates
index_endp_candidate = []
label = np.max(df_numpy[:, 3])

for i in range(label):
    index = np.argwhere(df_numpy[:, 3] == i+1)
    index_endp_candidate = np.append(index_endp_candidate, index[0], axis=0)
    index_endp_candidate = np.append(index_endp_candidate, index[-1], axis=0)

labels = np.arange(label)
labels = labels + 1

# endpoint estimation
endpoints = np.zeros(np.shape(data1))
patch_size = 1
start_end = []
endpoints_list = []

for i in range(np.shape(index_endp_candidate)[0]):

    index = index_endp_candidate[i].astype(np.int32)

    x = df_numpy[index, 0]
    y = df_numpy[index, 1]
    z = df_numpy[index, 2]
    label = df_numpy[index, 3]

    a = [x, y, z, label]
    start_end = np.append(start_end, a, axis=0)

    patch = centerline[x-patch_size:x+(patch_size+1),
            y-patch_size:y+(patch_size+1),
            z-patch_size:z+(patch_size+1)]
    other_label = np.delete(labels, label-1)

    endpoint_or_not = np.isin(patch, other_label)
    if True in endpoint_or_not:
        pass
    else:
        endpoints[x, y, z] = 1
        endpoints_list = np.append(endpoints_list, df_numpy[index, :])


endpoint_size = np.size(endpoints_list)
endpoint_size = int(endpoint_size/4)
endpoints_list = np.reshape(endpoints_list, [endpoint_size, 4])


# kernel = ball(2)
# endpoints_dilated = binary_dilation(endpoints, kernel)
# final = endpoints_dilated
#
# centerline_name = 'endpoint_' + niftiroot + '.nii.gz'
# mask = nib.Nifti1Image(final, img1.affine, img1.header)
# nib.save(mask, os.path.join(niftidir, centerline_name))


# co-occurrence matrix between branches

label = np.max(df_numpy[:, 3])

co_occurrence = np.zeros([label+1, label+1])
co_occurrence_inverse = np.zeros([label+1, label+1])

patch_size = 2
for j in range(2):
    for i in range(label):
        index = index_endp_candidate[(i*2)+(1*j)].astype(np.int32)
        x = df_numpy[index, 0]
        y = df_numpy[index, 1]
        z = df_numpy[index, 2]
        label = df_numpy[index, 3]

        patch = centerline[x-patch_size:x+(patch_size+1),
                y-patch_size:y+(patch_size+1),
                z-1:z+2]
        other_label = np.delete(labels, label-1)

        neighbour_label = np.isin(other_label, patch)
        a = np.where(neighbour_label == True)
        b = other_label[a]
        if j == 0:  # sometimes centerlines are inverse order. if top-to-down j =1, else j = 0
            co_occurrence_inverse[label, b] = 1
        else:
            co_occurrence[label, b] = 1


# calculate distance of candidate points from aorta
start_end = np.reshape(start_end, [label*2, 4])

filename1 = os.path.join(niftidir, initial_surface)
img1 = nib.load(filename1)
data1 = img1.get_fdata()

slice_index = np.argwhere(data1 > 0)
x = slice_index[0, 0]
y = slice_index[0, 1]
z = slice_index[0, 2]

# euclidean distance
distance_thresh = 30
aa = np.array([x, y, z])
bb = start_end[:, 0:3] - aa
cc = np.power(bb, 2)
dd = np.sum(cc, axis=1)
ee = np.sqrt(dd)

ff = np.argwhere(ee < distance_thresh)
ff = np.floor(ff/2)
labels_near_aorta = np.unique(ff)  # these labels should be join together as proxiaml branch close to aorta
labels_near_aorta = (labels_near_aorta).astype('int32')

# extracting main label and checking whether is not endpoint
flag = -1
labels_near_aorta_temp = labels_near_aorta
endpoints_label = endpoints_list[:, 3] - 1
while flag == -1:
    arg1 = np.argmax(ee[(labels_near_aorta_temp*2)+1])
    value1 = np.max(ee[(labels_near_aorta_temp*2)+1])

    arg2 = np.argmax(ee[(labels_near_aorta_temp*2)])
    value2 = np.max(ee[(labels_near_aorta_temp*2)])

    if value1 >= value2:
        arg = arg1
    else:
        arg = arg2

    label_proximal = labels_near_aorta_temp[arg]  # this label considered as main label close to aorta
    endpoint_or_not = np.isin(label_proximal, endpoints_label)
    if True in endpoint_or_not:
        labels_near_aorta_temp = labels_near_aorta_temp[labels_near_aorta_temp != label_proximal]
    else:
        flag = 1



# branch labeling right
STRUCT = {
    'Proximal RCA': -1,
    'Mid RCA': -1,
    'Distal RCA': -1,
    'Right PDA': -1,
    'V': -1,
    'AM': -1,
    'RPD': -1,
}

# level 1
STRUCT['Proximal RCA'] = [labels_near_aorta + 1]
previous_z = z

# level 2
degree_angle = 40

if STRUCT['Proximal RCA'] != -1:
    label = label_proximal + 1
    indx = np.argwhere(df_numpy[:, 3] == label)

    start_x_proximal = df_numpy[indx[0], 0]
    start_y_proximal = df_numpy[indx[0], 1]
    start_z_proximal = df_numpy[indx[0], 2]

    end_x_proximal = df_numpy[indx[-1], 0]
    end_y_proximal = df_numpy[indx[-1], 1]
    end_z_proximal = df_numpy[indx[-1], 2]

    # proximal_angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
    deltaY = end_y_proximal - start_y_proximal
    deltaX = end_x_proximal - start_x_proximal
    proximal_angle = atan2(deltaY, deltaX) * 180 / pi

    if (np.abs(end_z_proximal - previous_z) >= np.abs(start_z_proximal - previous_z)):
        neighbour = np.argwhere(co_occurrence[label, :] == 1)
        previous_z = end_z_proximal
    else:
        neighbour = np.argwhere(co_occurrence_inverse[label, :] == 1)
        previous_z = start_z_proximal

    num_neighbour = np.shape(neighbour)[0]
    angle_matrix = np.zeros([num_neighbour, 2])

    for i in range(num_neighbour):
        label = neighbour[i]
        label = label[0]
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]

        # angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal
        angle = atan2(deltaY, deltaX) * 180 / pi

        diff_angle = np.abs(proximal_angle - angle)

        angle_matrix[i, 0] = diff_angle
        angle_matrix[i, 1] = label

        if diff_angle <= degree_angle:
            STRUCT['Mid RCA'] = label
        else:
            STRUCT['V'] = label

    index = np.argsort(angle_matrix[:, 0], axis=0)

    if num_neighbour == 2 and (STRUCT['Mid RCA'] == -1 or STRUCT['V'] == -1):
        STRUCT['Mid RCA'] = angle_matrix[index[0], 1].astype('int64')
        STRUCT['V'] = angle_matrix[index[1], 1].astype('int64')

# level 3
degree_angle = 40

if STRUCT['Mid RCA'] != -1:
    label = STRUCT['Mid RCA']
    indx = np.argwhere(df_numpy[:, 3] == label)

    start_x_proximal = df_numpy[indx[0], 0]
    start_y_proximal = df_numpy[indx[0], 1]
    start_z_proximal = df_numpy[indx[0], 2]

    end_x_proximal = df_numpy[indx[-1], 0]
    end_y_proximal = df_numpy[indx[-1], 1]
    end_z_proximal = df_numpy[indx[-1], 2]

    # proximal_angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
    deltaY = end_y_proximal - start_y_proximal
    deltaX = end_x_proximal - start_x_proximal
    proximal_angle = atan2(deltaY, deltaX) * 180 / pi

    if (np.abs(end_z_proximal - previous_z) >= np.abs(start_z_proximal - previous_z)):
        neighbour = np.argwhere(co_occurrence[label, :] == 1)
        previous_z = end_z_proximal
    else:
        neighbour = np.argwhere(co_occurrence_inverse[label, :] == 1)
        previous_z = start_z_proximal

    num_neighbour = np.shape(neighbour)[0]
    angle_matrix = np.zeros([num_neighbour, 2])

    for i in range(num_neighbour):
        label = neighbour[i]
        label = label[0]
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]

        # angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal
        angle = atan2(deltaY, deltaX) * 180 / pi

        diff_angle = np.abs(proximal_angle - angle)

        angle_matrix[i, 0] = diff_angle
        angle_matrix[i, 1] = label

        if diff_angle <= degree_angle:
            STRUCT['AM'] = label
        else:
            STRUCT['Distal RCA'] = label

    index = np.argsort(angle_matrix[:, 0], axis=0)

    if num_neighbour == 2 and (STRUCT['AM'] == -1 or STRUCT['Distal RCA'] == -1):
        STRUCT['AM'] = angle_matrix[index[0], 1].astype('int64')
        STRUCT['Distal RCA'] = angle_matrix[index[1], 1].astype('int64')


# level 4
degree_angle = 40

if STRUCT['Distal RCA'] != -1:
    label = STRUCT['Distal RCA']
    indx = np.argwhere(df_numpy[:, 3] == label)

    start_x_proximal = df_numpy[indx[0], 0]
    start_y_proximal = df_numpy[indx[0], 1]
    start_z_proximal = df_numpy[indx[0], 2]

    end_x_proximal = df_numpy[indx[-1], 0]
    end_y_proximal = df_numpy[indx[-1], 1]
    end_z_proximal = df_numpy[indx[-1], 2]

    # proximal_angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
    deltaY = end_y_proximal - start_y_proximal
    deltaX = end_x_proximal - start_x_proximal
    proximal_angle = atan2(deltaY, deltaX) * 180 / pi

    if (np.abs(end_z_proximal - previous_z) >= np.abs(start_z_proximal - previous_z)):
        neighbour = np.argwhere(co_occurrence[label, :] == 1)
        previous_z = end_z_proximal
    else:
        neighbour = np.argwhere(co_occurrence_inverse[label, :] == 1)
        previous_z = start_z_proximal

    num_neighbour = np.shape(neighbour)[0]
    angle_matrix = np.zeros([num_neighbour, 2])

    for i in range(num_neighbour):
        label = neighbour[i]
        label = label[0]
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]

        # angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal
        angle = atan2(deltaY, deltaX) * 180 / pi

        diff_angle = np.abs(proximal_angle - angle)

        angle_matrix[i, 0] = diff_angle
        angle_matrix[i, 1] = label

        if diff_angle <= degree_angle:
            STRUCT['Right PDA'] = label
        else:
            STRUCT['RPD'] = label

    index = np.argsort(angle_matrix[:, 0], axis=0)

    if num_neighbour == 2 and (STRUCT['Right PDA'] == -1 or STRUCT['RPD'] == -1):
        STRUCT['Right PDA'] = angle_matrix[index[0], 1].astype('int64')
        STRUCT['RPD'] = angle_matrix[index[1], 1].astype('int64')

print(STRUCT)

# branch labeling left
STRUCT = {
    'Left main': -1,
    'Proximal LAD': -1,
    'Mid LAD': -1,
    'Distal LAD': -1,
    'First Diagonal': -1,
    'Second Diagonal': -1,
    'Proximal LCX': -1,
    'First marginal': -1,
    'Mid-distal LCX': -1,
    'Posterolateral branch': -1,
    'Left PDA': -1,
}

# level 1
STRUCT['Left main'] = [labels_near_aorta + 1]

# level 2
degree_angle = 40

if STRUCT['Left main'] != -1:
    label = label_proximal[0] + 1
    indx = np.argwhere(df_numpy[:, 3] == label)

    start_x_proximal = df_numpy[indx[0], 0]
    start_y_proximal = df_numpy[indx[0], 1]

    end_x_proximal = df_numpy[indx[-1], 0]
    end_y_proximal = df_numpy[indx[-1], 1]

    # proximal_angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
    deltaY = end_y_proximal - start_y_proximal
    deltaX = end_x_proximal - start_x_proximal
    proximal_angle = atan2(deltaY, deltaX) * 180 / pi

    neighbour = np.argwhere(co_occurrence[label, :] == 1)
    num_neighbour = np.shape(neighbour)[0]
    angle_matrix = np.zeros([num_neighbour, 2])

    for i in range(num_neighbour):
        label = neighbour[i]
        label = label[0]
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]

        # angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal
        angle = atan2(deltaY, deltaX) * 180 / pi

        diff_angle = np.abs(proximal_angle - angle)

        angle_matrix[i, 0] = diff_angle
        angle_matrix[i, 1] = label

        if diff_angle <= degree_angle:
            STRUCT['Proximal LAD'] = label
        else:
            STRUCT['Proximal LCX'] = label

    index = np.argsort(angle_matrix[:, 0], axis=0)

    if num_neighbour == 2 and (STRUCT['Proximal LAD'] == -1 or STRUCT['Proximal LCX'] == -1):
        STRUCT['Proximal LAD'] = angle_matrix[index[0], 1].astype('int64')
        STRUCT['Proximal LCX'] = angle_matrix[index[1], 1].astype('int64')

# level 3-1
degree_angle = 40

if STRUCT['Proximal LCX'] != -1:

    label = STRUCT['Proximal LCX']
    indx = np.argwhere(df_numpy[:, 3] == label)

    start_x_proximal = df_numpy[indx[0], 0]
    start_y_proximal = df_numpy[indx[0], 1]

    end_x_proximal = df_numpy[indx[-1], 0]
    end_y_proximal = df_numpy[indx[-1], 1]

    # proximal_angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
    deltaY = end_y_proximal - start_y_proximal
    deltaX = end_x_proximal - start_x_proximal
    proximal_angle = atan2(deltaY, deltaX) * 180 / pi

    neighbour = np.argwhere(co_occurrence[label, :] == 1)
    num_neighbour = np.shape(neighbour)[0]
    angle_matrix = np.zeros([num_neighbour, 2])

    for i in range(num_neighbour):
        label = neighbour[i]
        label = label[0]
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]

        # angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal
        angle = atan2(deltaY, deltaX) * 180 / pi

        diff_angle = np.abs(proximal_angle - angle)

        angle_matrix[i, 0] = diff_angle
        angle_matrix[i, 1] = label

        if diff_angle <= degree_angle:
            STRUCT['Mid-distal LCX'] = label
        else:
            STRUCT['First marginal'] = label

    index = np.argsort(angle_matrix[:, 0], axis=0)

    if num_neighbour == 2 and (STRUCT['Mid-distal LCX'] == -1 or STRUCT['First marginal'] == -1):
        STRUCT['Mid-distal LCX'] = angle_matrix[index[0], 1].astype('int64')
        STRUCT['First marginal'] = angle_matrix[index[1], 1].astype('int64')


# level 3-2
degree_angle = 20

if STRUCT['Proximal LAD'] != -1:

    label = STRUCT['Proximal LAD']
    indx = np.argwhere(df_numpy[:, 3] == label)

    start_x_proximal = df_numpy[indx[0], 0]
    start_y_proximal = df_numpy[indx[0], 1]

    end_x_proximal = df_numpy[indx[-1], 0]
    end_y_proximal = df_numpy[indx[-1], 1]

    # proximal_angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
    deltaY = end_y_proximal - start_y_proximal
    deltaX = end_x_proximal - start_x_proximal
    proximal_angle = atan2(deltaY, deltaX) * 180 / pi

    neighbour = np.argwhere(co_occurrence[label, :] == 1)
    num_neighbour = np.shape(neighbour)[0]
    angle_matrix = np.zeros([num_neighbour, 2])

    for i in range(num_neighbour):
        label = neighbour[i]
        label = label[0]
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]

        # angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal
        angle = atan2(deltaY, deltaX) * 180 / pi

        diff_angle = np.abs(proximal_angle - angle)

        angle_matrix[i, 0] = diff_angle
        angle_matrix[i, 1] = label

        if diff_angle <= degree_angle:
            STRUCT['Mid LAD'] = label
        else:
            STRUCT['First Diagonal'] = label

    index = np.argsort(angle_matrix[:, 0], axis=0)

    if num_neighbour == 2 and (STRUCT['Mid LAD'] == -1 or STRUCT['First Diagonal'] == -1):
        STRUCT['Mid LAD'] = angle_matrix[index[0], 1].astype('int64')
        STRUCT['First Diagonal'] = angle_matrix[index[1], 1].astype('int64')


# level 4-1
degree_angle = 20

if STRUCT['Mid-distal LCX'] != -1:

    label = STRUCT['Mid-distal LCX']
    indx = np.argwhere(df_numpy[:, 3] == label)

    start_x_proximal = df_numpy[indx[0], 0]
    start_y_proximal = df_numpy[indx[0], 1]

    end_x_proximal = df_numpy[indx[-1], 0]
    end_y_proximal = df_numpy[indx[-1], 1]

    # proximal_angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
    deltaY = end_y_proximal - start_y_proximal
    deltaX = end_x_proximal - start_x_proximal
    proximal_angle = atan2(deltaY, deltaX) * 180 / pi

    neighbour = np.argwhere(co_occurrence[label, :] == 1)
    num_neighbour = np.shape(neighbour)[0]
    angle_matrix = np.zeros([num_neighbour, 2])

    for i in range(num_neighbour):
        label = neighbour[i]
        label = label[0]

        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]

        # angle = getAngleBetweenPoints(start_x_proximal, start_y_proximal, end_x_proximal, end_y_proximal)
        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal
        angle = atan2(deltaY, deltaX) * 180 / pi

        diff_angle = np.abs(proximal_angle - angle)

        angle_matrix[i, 0] = diff_angle
        angle_matrix[i, 1] = label

        if diff_angle <= degree_angle:
            STRUCT['Left PDA'] = label
        else:
            STRUCT['Posterolateral branch'] = label

    index = np.argsort(angle_matrix[:, 0], axis=0)

    if num_neighbour == 2 and (STRUCT['Left PDA'] == -1 or STRUCT['Posterolateral branch'] == -1):
        STRUCT['Left PDA'] = angle_matrix[index[0], 1].astype('int64')
        STRUCT['Posterolateral branch'] = angle_matrix[index[1], 1].astype('int64')


a=1
# kernel = ball(3)
# dilated_centerline = centerline
# for i in range(label):
# # for i in range(14):
#     mask = centerline == (i+1)
#     mask = binary_dilation(mask, kernel)
#     dilated_centerline = np.where(mask > 0, i+1, dilated_centerline)




# kernel = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
#                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
#                   [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
# kernel = ball(1)
# convol = convolve(centerline, kernel, mode='constant', cval=0.0)




# start point and endpoint selection



myargs = 'vmtkbranchextractor -ifile vessel_aorta_centerline.vtp -ofile vessel_aorta_centerline_split.vtp'
mypype = pypes.PypeRun(myargs)


myargs = 'vmtksurfaceviewer -ifile vessel_aorta_centerline_split.vtp '
mypype = pypes.PypeRun(myargs)



a=1


# myargs = 'vmtknetworkextraction -levelsetstype "geodesic" -ifile img_mask.nii.gz -ofile levelset_img.nii.gz'
# mypype = pypes.PypeRun(myargs)




# myargs = 'vmtksurfacereader -ifile model_test.vtp -ofile centreline_test.vtp --pipe vmtkcenterlines --pipe vmtkrenderer ' \
#          '--pipe vmtksurfaceviewer -opacity 0.25' \
#          ' --pipe vmtksurfaceviewer -i @vmtkcenterlines.o -array MaximumInscribedSphereRadius'
# mypype = pypes.PypeRun(myargs)


# myargs = 'vmtkimagereader -ifile Original_data.nii.gz -ofile Original_data.vti'
# mypype = pypes.PypeRun(myargs)

# myargs = 'vmtkimageviewer -ifile C12N4_grayscale_crop.vti'
# mypype = pypes.PypeRun(myargs)
#
# myargs = 'vmtksurfaceviewer -ifile sections_along_centerline.vtp'
# mypype = pypes.PypeRun(myargs)

# myargs = 'vmtkimagecurvedmpr -ifile curvedMPR.vti -centerlinesfile centerlines.vtp'
# mypype = pypes.PypeRun(myargs)


# myargs = 'vmtkcenterlineresampling -length 0.01 -ifile C12N4_label_crop.ls2.mc.sm.cl.vtp --pipe vmtkcenterlineattributes' \
#          ' --pipe vmtkcenterlinegeometry --pipe vmtkimagecurvedmpr -ifile C12N4_grayscale_crop.vti -size 1000 ' \
#          '-spacing 0.01 -ofile reformat.vti --pipe vmtkimageviewer '
# mypype = pypes.PypeRun(myargs)


# myargs = 'vmtkcenterlineresampling -ifile centerlines.vtp --pipe vmtkcenterlineattributes ' \
#          '--pipe vmtkcenterlinegeometry --pipe vmtkimagecurvedmpr -ifile volumeTest2.mha ' \
#          '-ofile curvedMPR.vti --pipe vmtkimageviewer '
# mypype = pypes.PypeRun(myargs)


# making the sections along the centerline
myargs = 'vmtkcenterlineresampling -length 1 -ifile centreline_test.vtp -ofile resampled_centerline_test.vtp '
mypype = pypes.PypeRun(myargs)

myargs = 'vmtkcenterlinesections -ifile model_test.vtp -centerlinesfile resampled_centerline_test.vtp -ofile sections_along_centerline_test.vtp'
mypype = pypes.PypeRun(myargs)

myargs = 'vmtksurfaceviewer -ifile sections_along_centerline_test.vtp '
mypype = pypes.PypeRun(myargs)

myargs = 'vmtksurfacetonumpy -ifile centreline_test.vtp -ofile centerlinenumy.nii '
mypype = pypes.PypeRun(myargs)



# myargs = 'vmtkcenterlineattributes -ifile centreline_test.vtp -ofile foo_clat.vtp '
# mypype = pypes.PypeRun(myargs)
#
# myargs = 'vmtksurfaceviewer -ifile lumen_centerline.vtp'
# mypype = pypes.PypeRun(myargs)




a=1
# myargs = 'vmtkcenterlineresampling -length 0.01 -ifile centreline_test1.vtp -ofile resampled_centerline.vtp '
# mypype = pypes.PypeRun(myargs)
#
# myargs = 'vmtkcenterlinegeometry -ifile resampled_centerline.vtp --pipe vmtkcenterlineattributes' \
#          ' --pipe vmtkimagecurvedmpr -ifile lumen-heart-mask.nii -ofile sliced_image.vti '
# mypype = pypes.PypeRun(myargs)








# # cheking for  co-localized image and centerline
# myargs = 'vmtkrenderer --pipe vmtkimageviewer -ifile Original_data.vti --pipe vmtksurfaceviewer ' \
#          '-ifile centreline_test1.vtp -array MaximumInscribedSphereRadius'
# mypype = pypes.PypeRun(myargs)


#
# # ----------------------------------------------------------------------------------------------------------------------
# # LEVEL SET SEGMENTATION
# # ----------------------------------------------------------------------------------------------------------------------
#
# levelset_img = 'level_sets_' + niftiroot + '.nii.gz'
# myargs = 'vmtklevelsetsegmentation -ifile  ' + niftiname + '  -ofile ' + levelset_img
#
# if centreline_visualisation_only is not True:
#     mypype = pypes.PypeRun(myargs)
#
# # ----------------------------------------------------------------------------------------------------------------------
# # SURFACE MODELLING
# # ----------------------------------------------------------------------------------------------------------------------
#
# modelfile = 'model_' + niftiroot + '.vtp'
# myargs = 'vmtkmarchingcubes -ifile ' + levelset_img + ' -ofile ' + modelfile
#
# if centreline_visualisation_only is not True:
#     mypype = pypes.PypeRun(myargs)
#
# # ----------------------------------------------------------------------------------------------------------------------
# # CENTRELINE CALCULATION
# # ----------------------------------------------------------------------------------------------------------------------
#
# centreline_file = 'centreline_' + niftiroot + '.vtp'
# myargs = 'vmtkcenterlines -ifile ' + modelfile + ' -ofile ' + centreline_file
#
# if centreline_visualisation_only is not True:
#     mypype = pypes.PypeRun(myargs)
#
# # ----------------------------------------------------------------------------------------------------------------------
# # CENTRELINE VISUALISATION
# # ----------------------------------------------------------------------------------------------------------------------
#
# myargs = 'vmtksurfacereader -ifile ' + modelfile + ' --pipe ' + \
#          'vmtkrenderer --pipe ' + \
#          'vmtksurfaceviewer -opacity 0.25' + ' --pipe ' + \
#          'vmtksurfaceviewer -ifile ' + centreline_file + ' -array MaximumInscribedSphereRadius'
#
# mypype = pypes.PypeRun(myargs)
#
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Geometric Analysis
# # ----------------------------------------------------------------------------------------------------------------------
# curvedmpr_file = 'curvedmpr_' + niftiroot + '.vtp'
# myargs = 'vmtkimagecurvedmpr -ifile ' + niftiname + ' -centerlines ' + centreline_file + ' -ofile ' + curvedmpr_file
#
#
#
#
# if centreline_visualisation_only is not True:
#     mypype = pypes.PypeRun(myargs)
