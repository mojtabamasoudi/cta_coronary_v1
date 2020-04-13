import sys
import os
import numpy as np
import nibabel as nib
from skimage.morphology import ball, binary_dilation
import pandas as pd
from math import *

def centerline_calculation(niftidir, niftiname, niftiroot, niftiname_intersect):

    from vmtk import pypes
    from vmtk import vmtkscripts

    os.chdir(niftidir)

    # automatically centerline calculation

    surface = niftiroot + '.vtp'
    myargs = 'vmtkmarchingcubes -ifile ' + niftiname + ' -l 1.0 -ofile ' + surface
    mypype = pypes.PypeRun(myargs)

    smoothing_iteration = 2500
    surface_smooth = surface + '_smooth.vtp'
    myargs = 'vmtksurfacesmoothing -iterations ' + smoothing_iteration + ' -ifile ' + surface + ' -ofile ' + surface_smooth
    mypype = pypes.PypeRun(myargs)


    surface_intersect = niftiroot + '_intersect.vtp'
    myargs = 'vmtkmarchingcubes -ifile ' + niftiname_intersect + ' -l 1.0 -ofile ' + surface_intersect
    mypype = pypes.PypeRun(myargs)


    surface_clip = surface + '_clip.vtp'
    myargs = 'vmtksurfacecliploop -ifile ' + surface_smooth + \
             ' -i2file ' + surface_intersect + ' -ofile ' + surface_clip
    mypype = pypes.PypeRun(myargs)


    surface_centerline = surface + '_centerline.vtp'
    myargs = 'vmtknetworkextraction -ifile ' + surface_clip + ' -ofile ' + surface_centerline
    mypype = pypes.PypeRun(myargs)


    surface_centerline_sampling = surface + '_centerline_sampling.vtp'
    myargs = 'vmtklineresampling -ifile ' + surface_centerline + ' -length 0.1 -ofile ' + surface_centerline_sampling
    mypype = pypes.PypeRun(myargs)


    myargs = 'vmtksurfacereader -ifile ' + surface_clip + ' --pipe ' + \
             'vmtkrenderer --pipe ' + \
             'vmtksurfaceviewer -opacity 0.25' + ' --pipe ' + \
             'vmtksurfaceviewer -ifile ' + surface_centerline_sampling + ' -array MaximumInscribedSphereRadius'

    mypype = pypes.PypeRun(myargs)


    # export senterline coordinates

    centerlineReader = vmtkscripts.vmtkSurfaceReader()
    centerlineReader.InputFileName = surface_centerline_sampling
    centerlineReader.Execute()

    clNumpyAdaptor = vmtkscripts.vmtkCenterlinesToNumpy()
    clNumpyAdaptor.Centerlines = centerlineReader.Surface
    clNumpyAdaptor.Execute()

    numpyCenterlines = clNumpyAdaptor.ArrayDict
    points = numpyCenterlines['Points']

    labels = numpyCenterlines['CellData']
    cell_points = labels['CellPointIds']

    for i in range(np.shape(cell_points)[0]):
        number = np.shape(cell_points[i])[0]
        index = np.ones(number) * (i + 1)
        if i == 0:
            topology_labels = index
        else:
            topology_labels = np.append(topology_labels, index, axis=0)

    myargs = 'vmtkimagereader -ifile  ' + niftiname
    mypype = pypes.PypeRun(myargs)
    RasToIjkMatrixCoefficients = mypype.GetScriptObject('vmtkimagereader', '0').RasToIjkMatrixCoefficients
    XyzToRasMatrixCoefficients = mypype.GetScriptObject('vmtkimagereader', '0').XyzToRasMatrixCoefficients
    ras2ijk = np.asarray(RasToIjkMatrixCoefficients)
    ras2ijk = np.reshape(ras2ijk, (4, 4))
    xyz2ras = np.asarray(XyzToRasMatrixCoefficients)
    xyz2ras = np.reshape(xyz2ras, (4, 4))

    for x in range(points.shape[0]):
        xyz = points[x]
        xyz = np.append(xyz, 1)
        ras = np.dot(xyz2ras, xyz)
        ijk = np.dot(ras2ijk, ras)
        ijk[3] = topology_labels[x]
        if x == 0:
            points_ijk = ijk
        else:
            points_ijk = np.append(points_ijk, ijk, axis=0)

    points_ijk = np.reshape(points_ijk, [x + 1, 4])
    points_ijk = np.round(points_ijk[:, 0:4])
    print(points_ijk)

    csv_file = 'centreline_' + niftiroot + '.csv'
    np.savetxt(csv_file, points_ijk, delimiter=",")

    return csv_file


def endpoint_calculation(niftidir, size_data, csv_file, intersect):

    centerlines = np.zeros(size_data)
    endpoints = np.zeros(size_data)

    filename1 = os.path.join(niftidir, csv_file)
    df = pd.read_csv(filename1)
    df_numpy = np.round(df.to_numpy().astype(np.int32))

    for i in range(df.values.shape[0]):
        x = df_numpy[i, 0]
        y = df_numpy[i, 1]
        z = df_numpy[i, 2]
        label = df_numpy[i, 3]
        centerlines[x, y, z] = label

    # save centerline in niffti format
    centerline_name = 'centerlines_' + intersect
    filename1 = os.path.join(niftidir, intersect)
    img1 = nib.load(filename1)
    mask = nib.Nifti1Image(centerlines, img1.affine, img1.header)
    nib.save(mask, os.path.join(niftidir, centerline_name))

    # endpoint candidates
    index_endp_candidate = []
    label = np.max(df_numpy[:, 3])

    for i in range(label):
        index = np.argwhere(df_numpy[:, 3] == i + 1)
        index_endp_candidate = np.append(index_endp_candidate, index[0], axis=0)
        index_endp_candidate = np.append(index_endp_candidate, index[-1], axis=0)

    labels = np.arange(label)
    labels = labels + 1

    # endpoints estimation
    endpoints = np.zeros(size_data)
    patch_size = 2
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

        patch = centerlines[x - patch_size:x + (patch_size + 1),
                y - patch_size:y + (patch_size + 1),
                z - patch_size:z + (patch_size + 1)]
        other_label = np.delete(labels, label - 1)

        endpoint_or_not = np.isin(patch, other_label)
        if True in endpoint_or_not:
            pass
        else:
            endpoints[x, y, z] = 1
            endpoints_list = np.append(endpoints_list, df_numpy[index, :])

    endpoint_size = np.size(endpoints_list)
    endpoint_size = int(endpoint_size / 4)
    endpoints_list = np.reshape(endpoints_list, [endpoint_size, 4])

    # save csv file endpoint coordinates
    csv_endpoint_file = 'endpoint_' + csv_file
    np.savetxt(csv_endpoint_file, endpoints_list, delimiter=",")

    # # save endpoints in niffti format
    # centerline_name = 'endpoints_.nii.gz'
    # filename1 = os.path.join(niftidir, intersect)
    # img1 = nib.load(filename1)
    # mask = nib.Nifti1Image(endpoints, img1.affine, img1.header)
    # nib.save(mask, os.path.join(niftidir, centerline_name))

    # co-occurrence matrix between branches (shows neighbourhood for each branches)
    label = np.max(df_numpy[:, 3])

    # in some cases the direction of startpoint and endpoint are not forward. For this cases we use
    # co_occurrence_inverse matrix

    co_occurrence = np.zeros([label + 1, label + 1])
    co_occurrence_inverse = np.zeros([label + 1, label + 1])

    patch_size = 2
    for j in range(2):
        for i in range(label):
            index = index_endp_candidate[(i * 2) + (1 * j)].astype(np.int32)
            x = df_numpy[index, 0]
            y = df_numpy[index, 1]
            z = df_numpy[index, 2]
            label = df_numpy[index, 3]

            patch = centerlines[x - patch_size:x + (patch_size + 1),
                    y - patch_size:y + (patch_size + 1),
                    z - 1:z + 2]
            other_label = np.delete(labels, label - 1)

            neighbour_label = np.isin(other_label, patch)
            a = np.where(neighbour_label == True)
            b = other_label[a]
            if j == 0:  # sometimes centerlines are inverse order. if top-to-down j =1, else j = 0
                co_occurrence_inverse[label, b] = 1
            else:
                co_occurrence[label, b] = 1

    # calculate distance of candidate points from aorta
    start_end = np.reshape(start_end, [label * 2, 4])

    filename1 = os.path.join(niftidir, intersect)
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
    ff = np.floor(ff / 2)
    labels_near_aorta = np.unique(ff)  # these labels should be join together as proximal branch close to aorta
    labels_near_aorta = (labels_near_aorta).astype('int32')

    # extracting main label and checking whether is endpoint or not
    # if was endpoint, try another big one
    flag = -1
    labels_near_aorta_temp = labels_near_aorta
    endpoints_label = endpoints_list[:, 3] - 1
    while flag == -1:
        arg1 = np.argmax(ee[(labels_near_aorta_temp * 2) + 1])
        value1 = np.max(ee[(labels_near_aorta_temp * 2) + 1])

        arg2 = np.argmax(ee[(labels_near_aorta_temp * 2)])
        value2 = np.max(ee[(labels_near_aorta_temp * 2)])

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

    return co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal, z


def branch_labeling_left(niftidir, csv_file, co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal,
                         aorta_z):

    filename1 = os.path.join(niftidir, csv_file)
    df = pd.read_csv(filename1)
    df_numpy = np.round(df.to_numpy().astype(np.int32))

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
        label = label_proximal + 1
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]

        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal

        # calculation angle between branches and horizontal axes
        proximal_angle = atan2(deltaY, deltaX) * 180 / pi

        # finding neighbour of this branch
        neighbour = np.argwhere(co_occurrence[label, :] == 1)
        num_neighbour = np.shape(neighbour)[0]
        angle_matrix = np.zeros([num_neighbour, 2])

        # comparing branch angle and its neighbour angle
        for i in range(num_neighbour):
            label = neighbour[i]
            label = label[0]
            indx = np.argwhere(df_numpy[:, 3] == label)

            start_x_proximal = df_numpy[indx[0], 0]
            start_y_proximal = df_numpy[indx[0], 1]

            end_x_proximal = df_numpy[indx[-1], 0]
            end_y_proximal = df_numpy[indx[-1], 1]

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

        # decision for some branches that both neighbour angles are bigger or smaller than proximal angle
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

    print(STRUCT)

    return STRUCT


def branch_labeling_right(niftidir, csv_file, co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal,
                          aorta_z):

    filename1 = os.path.join(niftidir, csv_file)
    df = pd.read_csv(filename1)
    df_numpy = np.round(df.to_numpy().astype(np.int32))

    degree_angle = 40

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
    previous_z = aorta_z

    # level 2

    if STRUCT['Proximal RCA'] != -1:
        label = label_proximal + 1
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]
        start_z_proximal = df_numpy[indx[0], 2]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]
        end_z_proximal = df_numpy[indx[-1], 2]

        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal
        proximal_angle = atan2(deltaY, deltaX) * 180 / pi

        # decision for using co_occurrence or co_occurrence_inverse matrix to calculate neighbour
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

    if STRUCT['Mid RCA'] != -1:
        label = STRUCT['Mid RCA']
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]
        start_z_proximal = df_numpy[indx[0], 2]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]
        end_z_proximal = df_numpy[indx[-1], 2]

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

    if STRUCT['Distal RCA'] != -1:
        label = STRUCT['Distal RCA']
        indx = np.argwhere(df_numpy[:, 3] == label)

        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]
        start_z_proximal = df_numpy[indx[0], 2]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]
        end_z_proximal = df_numpy[indx[-1], 2]

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

    return STRUCT


def main():

    niftidir = r'D:\Mojtaba\Dataset_test\9'

    aorta_name = 'aorta.nii.gz'
    kernel_size = 2  # dilation for connecting aorta and arteries

    niftiname = 'coronary.nii.gz'

    niftiname_L = 'L_coronary.nii.gz'
    niftiroot_L = 'L_coronary'

    niftiname_R = 'R_coronary.nii.gz'
    niftiroot_R = 'R_coronary'

    # left and right seperation
    filename1 = os.path.join(niftidir, niftiname)
    img1 = nib.load(filename1)
    coronary = img1.get_fdata()

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

    kernel = ball(kernel_size)
    dilated_aorta = binary_dilation(aorta, kernel)

    # Left side

    intersect = dilated_left_cor * dilated_aorta

    slice_index = np.argwhere(intersect == 1)
    two_surface = np.zeros(np.shape(intersect))
    x = slice_index[0, 0]
    y = slice_index[0, 1]
    z = slice_index[0, 2]

    kernel = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                       [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                       [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

    two_surface[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = kernel
    two_surface[x - 1:x + 2, y - 1:y + 2, z:z + 3] = kernel

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

    two_surface[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = kernel
    two_surface[x - 1:x + 2, y - 1:y + 2, z:z + 3] = kernel

    mask = nib.Nifti1Image(two_surface, img1.affine, img1.header)

    intersect_R = niftiroot_R +'_intersect.nii.gz'
    nib.save(mask, os.path.join(niftidir, intersect_R))

    # calculation left side centerline
    csv_file_L = centerline_calculation(niftidir, niftiname_L, niftiroot_L, intersect_L)

    # calculation right side centerline
    csv_file_R = centerline_calculation(niftidir, niftiname_R, niftiroot_R, intersect_R)

    # calculation of endpoints, co_occurrence matrix and labels near to aorta
    # left side
    co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal, aorta_z = endpoint_calculation\
        (niftidir, np.shape(coronary), csv_file_L, intersect_L)

    # labeling branches
    # left side
    struct = branch_labeling_left(niftidir, csv_file_L, co_occurrence, co_occurrence_inverse,
                                  labels_near_aorta, label_proximal, aorta_z)

    # calculation of endpoints, co_occurrence matrix and labels near to aorta
    # right side
    co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal, aorta_z = endpoint_calculation\
        (niftidir, np.shape(coronary), csv_file_R, intersect_R)

    # labeling branches
    # right side
    struct = branch_labeling_right(niftidir, csv_file_R, co_occurrence, co_occurrence_inverse,
                                  labels_near_aorta, label_proximal, aorta_z)



if __name__ == '__main__':
    main()


