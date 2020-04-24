import sys
import os
import numpy as np
import nibabel as nib
from skimage.morphology import ball, binary_dilation
import pandas as pd
from math import *
import time

def centerline_calculation(niftidir, niftiname, niftiroot, niftiname_intersect):

    from vmtk import pypes
    from vmtk import vmtkscripts

    os.chdir(niftidir)

    # automatically centerline calculation

    surface = niftiroot + '.vtp'
    myargs = 'vmtkmarchingcubes -ifile ' + niftiname + ' -l 1.0 -ofile ' + surface
    mypype = pypes.PypeRun(myargs)

    smoothing_iteration = 2500
    surface_smooth = niftiroot + '_smooth.vtp'
    myargs = 'vmtksurfacesmoothing -iterations ' + str(smoothing_iteration) + ' -ifile ' + surface + ' -ofile ' + surface_smooth
    mypype = pypes.PypeRun(myargs)


    surface_intersect = niftiroot + '_intersect.vtp'
    myargs = 'vmtkmarchingcubes -ifile ' + niftiname_intersect + ' -l 1.0 -ofile ' + surface_intersect
    mypype = pypes.PypeRun(myargs)


    surface_clip = niftiroot + '_clip.vtp'
    myargs = 'vmtksurfacecliploop -ifile ' + surface_smooth + \
             ' -i2file ' + surface_intersect + ' -ofile ' + surface_clip
    mypype = pypes.PypeRun(myargs)


    surface_centerline = niftiroot + '_centerline.vtp'
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


def endpoint_calculation(niftidir, size_data, csv_file, niftiroot, intersect):

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
    centerline_name = niftiroot + '_centerlines.nii.gz'
    filename1 = os.path.join(niftidir, intersect)
    img1 = nib.load(filename1)
    data1 = img1.get_fdata()
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
    branch_label_map = np.ones([np.shape(df_numpy)[0], 1]) * -1
    branch_parenthood = np.ones([np.shape(df_numpy)[0], 1]) * -1

    # branch labeling left
    STRUCT = {
        'aorta' : -1, # 0
        'Left main': -1,  # 5
        'Proximal LAD': -1,  # 6
        'Mid LAD': -1,  # 7
        'Distal LAD': -1,  # 8
        'First Diagonal': -1,  # 9
        'Second Diagonal': -1,  # 10
        'Proximal LCX': -1,  # 11
        'First marginal': -1,  # 12
        'Mid-distal LCX': -1,  # 13
        'Posterolateral branch': -1,  # 14
        'Left PDA': -1,  # 15
    }

    # level 1
    STRUCT['aorta'] = [labels_near_aorta + 1]

    # set label zero to all labels near to aorta
    num_labels = np.shape(labels_near_aorta)[0]
    for i in range(num_labels):
        label = STRUCT['aorta'][0][i]
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 0
        branch_parenthood[index] = 0

    label = label_proximal + 1
    STRUCT['Left main'] = label
    index = np.argwhere(df_numpy[:, 3] == label)
    branch_label_map[index] = 5
    branch_parenthood[index] = 0



    # level 2
    degree_angle = 40

    if STRUCT['Left main'] != -1:

        label = STRUCT['Left main']
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

        label = STRUCT['Proximal LAD']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 6
        branch_parenthood[index] = label_proximal + 1

        label = STRUCT['Proximal LCX']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 11
        branch_parenthood[index] = label_proximal + 1


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

        label = STRUCT['Mid-distal LCX']
        parenthood = STRUCT['Proximal LCX']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 13
        branch_parenthood[index] = parenthood

        label = STRUCT['First marginal']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 12
        branch_parenthood[index] = parenthood




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

        label = STRUCT['Mid LAD']
        parenthood = STRUCT['Proximal LAD']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 7
        branch_parenthood[index] = parenthood

        label = STRUCT['First Diagonal']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 9
        branch_parenthood[index] = parenthood

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

        label = STRUCT['Left PDA']
        parenthood = STRUCT['Mid-distal LCX']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 15
        branch_parenthood[index] = parenthood

        label = STRUCT['Posterolateral branch']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 14
        branch_parenthood[index] = parenthood


    print(STRUCT)

    output = np.ones([np.shape(df_numpy)[0], np.shape(df_numpy)[1] + 2])
    output[:, 0:4] = df_numpy
    output[:, 4:5] = branch_label_map
    output[:, 5:6] = branch_parenthood

    csv_file = 'branch_' + csv_file
    np.savetxt(csv_file, output, delimiter=",")

    # convert to JSON file
    import json

    # branch_name = 'aorta'
    # if STRUCT[branch_name] != -1:
    #     label = list(STRUCT[branch_name][0])
    #     label = list(map(int, label))
    #     parenthood = 0
    #     label_map = 0
    #     json_string = dict_to_json(branch_name, parenthood, label, label_map, output)

    branch_name = 'Left main'
    if STRUCT[branch_name] != -1:
        parenthood = 0
        label_map = 5
        json_string = dict_to_json(branch_name, parenthood, label_map, output)

    branch_name = 'Proximal LAD'
    if STRUCT[branch_name] != -1:
        parenthood = 5
        label_map = 6
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Mid LAD'
    if STRUCT[branch_name] != -1:
        parenthood = 6
        label_map = 7
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Distal LAD'
    if STRUCT[branch_name] != -1:
        parenthood = 7
        label_map = 8
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'First Diagonal'
    if STRUCT[branch_name] != -1:
        parenthood = 6
        label_map = 9
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Second Diagonal'
    if STRUCT[branch_name] != -1:
        parenthood = 7
        label_map = 10
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Proximal LCX'
    if STRUCT[branch_name] != -1:
        parenthood = 5
        label_map = 11
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Mid-distal LCX'
    if STRUCT[branch_name] != -1:
        parenthood = 11
        label_map = 13
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'First marginal'
    if STRUCT[branch_name] != -1:
        parenthood = 11
        label_map = 12
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Posterolateral branch'
    if STRUCT[branch_name] != -1:
        parenthood = 13
        label_map = 14
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Left PDA'
    if STRUCT[branch_name] != -1:
        parenthood = 13
        label_map = 15
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    # for branches without label
    branch_name = 'unknown'
    parenthood = -1
    label_map = -1
    json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
    json_string = json_string + json_string1

    os.chdir(niftidir)
    filename = csv_file + '.txt'
    with open(filename, 'w') as outfile:
        json.dump(json_string, outfile)


def dict_to_json(branch_name, parenthood, label_map, output):

    import json
    index = np.argwhere(output[:, 4] == label_map)
    coordinate_dict = list()
    for i in range(np.shape(index)[0]):
        dict_temp = dict(X=int(output[index[i], 0][0]), Y=int(output[index[i], 1][0]),
                         Z=int(output[index[i], 2][0]))
        coordinate_dict.append(dict_temp)

    coordinate_dict = tuple(coordinate_dict)

    dict_object = dict(branch_name=branch_name, parenthood=parenthood, label=label_map, coordinates=coordinate_dict)
    json_string = json.dumps(dict_object)

    return json_string


def branch_labeling_right(niftidir, csv_file, co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal,
                          aorta_z):

    filename1 = os.path.join(niftidir, csv_file)
    df = pd.read_csv(filename1)
    df_numpy = np.round(df.to_numpy().astype(np.int32))
    branch_label_map = np.ones([np.shape(df_numpy)[0], 1]) * -1
    branch_parenthood = np.ones([np.shape(df_numpy)[0], 1]) * -1

    degree_angle = 40

    # branch labeling right
    STRUCT = {
        'aorta': -1, # 0
        'Proximal RCA': -1,  # 1
        'Mid RCA': -1,  # 2
        'Distal RCA': -1,  # 3
        'Right PDA': -1,  # 4
        'V': -1,  # 16
        'AM': -1,  # 17
        'RPD': -1,  # 18
    }

    # level 1
    STRUCT['Proximal RCA'] = [labels_near_aorta + 1]
    previous_z = aorta_z

    # set label 1 to all labels left main
    num_labels = np.shape(labels_near_aorta)[0]
    for i in range(num_labels):
        label = STRUCT['Proximal RCA'][0][i]
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 0
        branch_parenthood[index] = 0

    label = label_proximal + 1
    STRUCT['Proximal RCA'] = label
    index = np.argwhere(df_numpy[:, 3] == label)
    branch_label_map[index] = 1
    branch_parenthood[index] = 0


    # level 2

    if STRUCT['Proximal RCA'] != -1:

        label = STRUCT['Proximal RCA']
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

        label = STRUCT['Mid RCA']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 2
        branch_parenthood[index] = label_proximal + 1

        label = STRUCT['V']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 16
        branch_parenthood[index] = label_proximal + 1


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

        label = STRUCT['AM']
        parenthood = STRUCT['Mid RCA']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 17
        branch_parenthood[index] = parenthood

        label = STRUCT['Distal RCA']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 3
        branch_parenthood[index] = parenthood


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

        label = STRUCT['Right PDA']
        parenthood = STRUCT['Distal RCA']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 4
        branch_parenthood[index] = parenthood

        label = STRUCT['RPD']
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 18
        branch_parenthood[index] = parenthood

    print(STRUCT)

    output = np.ones([np.shape(df_numpy)[0], np.shape(df_numpy)[1] + 2])
    output[:, 0:4] = df_numpy
    output[:, 4:5] = branch_label_map
    output[:, 5:6] = branch_parenthood

    csv_file = 'branch_' + csv_file
    np.savetxt(csv_file, output, delimiter=",")

    # convert to JSON file
    import json

    # branch_name = 'aorta'
    # if STRUCT[branch_name] != -1:
    #     label = list(STRUCT[branch_name][0])
    #     label = list(map(int, label))
    #     parenthood = 0
    #     label_map = 0
    #     json_string = dict_to_json(branch_name, parenthood, label, label_map, output)

    branch_name = 'Proximal RCA'
    if STRUCT[branch_name] != -1:
        parenthood = 0
        label_map = 1
        json_string = dict_to_json(branch_name, parenthood, label_map, output)
        # json_string = json_string + json_string1

    branch_name = 'Mid RCA'
    if STRUCT[branch_name] != -1:
        parenthood = 1
        label_map = 2
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Distal RCA'
    if STRUCT[branch_name] != -1:
        parenthood = 2
        label_map = 3
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'Right PDA'
    if STRUCT[branch_name] != -1:
        parenthood = 3
        label_map = 4
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'V'
    if STRUCT[branch_name] != -1:
        parenthood = 1
        label_map = 16
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'AM'
    if STRUCT[branch_name] != -1:
        parenthood = 2
        label_map = 17
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    branch_name = 'RPD'
    if STRUCT[branch_name] != -1:
        parenthood = 3
        label_map = 18
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1

    # for branches without label
    branch_name = 'unknown'
    parenthood = -1
    label_map = -1
    json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
    json_string = json_string + json_string1


    os.chdir(niftidir)
    filename = csv_file + '.txt'
    with open(filename, 'w') as outfile:
        json.dump(json_string, outfile)


def cmpr_preparation(niftidir, niftiroot, dilated_cor_aorta_name, csv_endpoint_file, cx, cy, cz ):

    from vmtk import pypes
    from vmtk import vmtkscripts

    os.chdir(niftidir)

    # automatically centerline calculation

    surface = niftiroot + '_cor_aorta.vtp'
    myargs = 'vmtkmarchingcubes -ifile ' + dilated_cor_aorta_name + ' -l 1.0 -ofile ' + surface
    mypype = pypes.PypeRun(myargs)

    smoothing_iteration = 2500
    surface_smooth = niftiroot + '_smooth.vtp'
    myargs = 'vmtksurfacesmoothing -iterations ' + str(smoothing_iteration) + ' -ifile ' + surface + ' -ofile ' + surface_smooth
    mypype = pypes.PypeRun(myargs)


    filename1 = os.path.join(niftidir, csv_endpoint_file)
    df = pd.read_csv(filename1)
    df_numpy = np.round(df.to_numpy().astype(np.int32))

    for i in range(df.values.shape[0]):
        x = df_numpy[i, 0]
        y = df_numpy[i, 1]
        z = df_numpy[i, 2]
        label = df_numpy[i, 3]

        centreline_file = niftiroot + '_centerline_' + str(i) + '.vtp'

        SourcePoint = [cx, cy, cz]
        TargetPoint = [x, y, z]

        modelfile = surface_smooth

        myargs = 'vmtkimagereader -ifile  ' + dilated_cor_aorta_name
        mypype = pypes.PypeRun(myargs)

        RasToIjkMatrixCoefficients = mypype.GetScriptObject('vmtkimagereader', '0').RasToIjkMatrixCoefficients
        XyzToRasMatrixCoefficients = mypype.GetScriptObject('vmtkimagereader', '0').XyzToRasMatrixCoefficients
        ras2ijk = np.asarray(RasToIjkMatrixCoefficients)
        ras2ijk = np.reshape(ras2ijk, (4, 4))
        xyz2ras = np.asarray(XyzToRasMatrixCoefficients)
        xyz2ras = np.reshape(xyz2ras, (4, 4))

        Image = mypype.GetScriptObject('vmtkimagereader', '0').Image

        print([SourcePoint[0], SourcePoint[1], SourcePoint[2]])
        sourcepoint_img = Image.GetPoint(
            Image.ComputePointId([int(SourcePoint[0]), int(SourcePoint[1]), int(SourcePoint[2])]))
        print(sourcepoint_img)

        # sourcepoint_orig = Image.ComputeStructuredCoordinates(Image.FindPoint(Image.GetPoint(Image.ComputePointId([SourcePoint[0], SourcePoint[1], SourcePoint[2]]))))
        # sourcepoint_orig = Image.GetPoint([SourcePoint[0], SourcePoint[1], SourcePoint[2]])
        # print(sourcepoint_orig)

        # ##################### DEBUG #####################
        # import code
        # code.interact(local=locals())  # inline debugging
        # #################################################

        targetpoint_img = Image.GetPoint(
            Image.ComputePointId([int(TargetPoint[0]), int(TargetPoint[1]), int(TargetPoint[2])]))
        print(targetpoint_img)

        myargs = 'vmtksurfacereader -ifile  ' + modelfile
        mypype = pypes.PypeRun(myargs)
        Surface = mypype.GetScriptObject('vmtksurfacereader', '0').Surface

        # sourcepointId_surf = str(Surface.FindPoint(Surface.GetPoint(Surface.FindPoint(sourcepoint_img))))
        sourcepointId_surf = str(Surface.FindPoint(sourcepoint_img))
        print(sourcepointId_surf)

        # targetpointId_surf = str(Surface.FindPoint(Surface.GetPoint(Surface.FindPoint(targetpoint_img))))
        targetpointId_surf = str(Surface.FindPoint(targetpoint_img))
        print(targetpointId_surf)

        # import sys
        # sys.exit(0)

        # centreline_file = 'centreline_' + niftiroot + '.vtp'
        # sourcepoint = '197 23 9'
        # targetpoint = '195 25 53'
        # myargs = 'vmtkcenterlines -ifile ' + modelfile + ' -ofile ' + centreline_file + ' -sourcepoints ' + sourcepoint_surf + ' -targetpoints ' + targetpoint_surf + ' -seedselector pointlist'
        myargs = 'vmtkcenterlines -ifile ' + modelfile + ' -ofile ' + centreline_file + ' -sourceids ' + sourcepointId_surf + ' -targetids ' + targetpointId_surf + ' -seedselector idlist'
        mypype = pypes.PypeRun(myargs)

        myargs = 'vmtksurfacereader -ifile ' + surface_smooth + ' --pipe ' + \
                 'vmtkrenderer --pipe ' + \
                 'vmtksurfaceviewer -opacity 0.25' + ' --pipe ' + \
                 'vmtksurfaceviewer -ifile ' + centreline_file + ' -array MaximumInscribedSphereRadius'

        mypype = pypes.PypeRun(myargs)

        # export senterline coordinates
        centerlineReader = vmtkscripts.vmtkSurfaceReader()
        centerlineReader.InputFileName = centreline_file
        centerlineReader.Execute()

        clNumpyAdaptor = vmtkscripts.vmtkCenterlinesToNumpy()
        clNumpyAdaptor.Centerlines = centerlineReader.Surface
        # clNumpyAdaptor.ConvertCellToPoint = 1
        clNumpyAdaptor.Execute()

        numpyCenterlines = clNumpyAdaptor.ArrayDict
        points = numpyCenterlines['Points']

        # ##################### DEBUG #####################
        # import code
        # code.interact(local=locals())  # inline debugging
        # #################################################

        for x in range(points.shape[0]):
            xyz = points[x]
            xyz = np.append(xyz, 1)
            ras = np.dot(xyz2ras, xyz)
            ijk = np.dot(ras2ijk, ras)
            if x == 0:
                points_ijk = ijk
            else:
                points_ijk = np.append(points_ijk, ijk, axis=0)

        points_ijk = np.reshape(points_ijk, [x + 1, 4])
        points_ijk = np.round(points_ijk[:, 0:3])
        print(points_ijk)

        csv_file = niftiroot + '_centerline_' + str(i) + '.csv'
        np.savetxt(csv_file, points_ijk, delimiter=",")



def main():
    tic = time.time()
    niftidir = r'D:\Mojtaba\Dataset_test\test\002_bi_shuying_ct1668532'

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

    # kernel = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    size_surface = 4
    kernel = np.ones([size_surface*2+1, size_surface*2+1, size_surface*2+1])


    two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
    z - size_surface:z + (size_surface+1)] = kernel
    two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
    z - (size_surface-1):z + (size_surface+2)] = kernel

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

    two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
    z - size_surface:z + (size_surface+1)] = kernel
    two_surface[x - size_surface:x + (size_surface+1), y - size_surface:y + (size_surface+1),
    z - (size_surface-1):z + (size_surface+2)] = kernel

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
    # right side
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


