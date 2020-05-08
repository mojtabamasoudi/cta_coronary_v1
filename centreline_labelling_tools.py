import os
import numpy as np
import nibabel as nib
from math import *
from numpy import genfromtxt


def centerline_calculation(niftidir, niftiname, niftiroot, niftiname_intersect):
    from vmtk import pypes
    from vmtk import vmtkscripts

    os.chdir(niftidir)

    # automatically centerline calculation

    surface = niftiroot + '.vtp'
    myargs = 'vmtkmarchingcubes -ifile ' + niftiname + ' -l 1.0 -ofile ' + surface
    mypype = pypes.PypeRun(myargs)

    smoothing_iteration = 2700
    surface_smooth = niftiroot + '_smooth.vtp'
    myargs = 'vmtksurfacesmoothing -iterations ' + str(
        smoothing_iteration) + ' -ifile ' + surface + ' -ofile ' + surface_smooth
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

    # export centerline coordinates

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
    df_numpy = genfromtxt(filename1, delimiter=',')
    df_numpy = df_numpy.astype(np.int32)

    if df_numpy.shape.__len__() == 1:
        df_numpy_size = 1
    else:
        df_numpy_size = df_numpy.shape[0]

    for i in range(df_numpy_size):

        if df_numpy_size == 1:
            x = df_numpy[0]
            y = df_numpy[1]
            z = df_numpy[2]
            label = df_numpy[3]
        else:
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

    intersect_coordinates = np.array([x, y, z])

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

    flag = -1
    labels_near_aorta_temp = labels_near_aorta
    endpoints_label = endpoints_list[:, 3] - 1

    # extracting main label
    arg1 = np.argmax(ee[(labels_near_aorta_temp * 2) + 1])
    value1 = np.max(ee[(labels_near_aorta_temp * 2) + 1])

    arg2 = np.argmax(ee[(labels_near_aorta_temp * 2)])
    value2 = np.max(ee[(labels_near_aorta_temp * 2)])

    if value1 >= value2:
        arg = arg1
    else:
        arg = arg2

    label_proximal = labels_near_aorta_temp[arg]  # this label considered as main label close to aorta

    # extracting main label and checking whether is endpoint or not
    # if was endpoint, try another big one
    # this only works when direction of start_end point was correct

    # while flag == -1:
    #     arg1 = np.argmax(ee[(labels_near_aorta_temp * 2) + 1])
    #     value1 = np.max(ee[(labels_near_aorta_temp * 2) + 1])
    #
    #     arg2 = np.argmax(ee[(labels_near_aorta_temp * 2)])
    #     value2 = np.max(ee[(labels_near_aorta_temp * 2)])
    #
    #     if value1 >= value2:
    #         arg = arg1
    #     else:
    #         arg = arg2
    #
    #     label_proximal = labels_near_aorta_temp[arg]  # this label considered as main label close to aorta
    #     endpoint_or_not = np.isin(label_proximal, endpoints_label)
    #     if True in endpoint_or_not:
    #         labels_near_aorta_temp = labels_near_aorta_temp[labels_near_aorta_temp != label_proximal]
    #     else:
    #         flag = 1

    # delete endpoint near to aorta
    distance_thresh = 30
    aa = np.array([x, y, z])
    bb = endpoints_list[:, 0:3] - aa
    cc = np.power(bb, 2)
    dd = np.sum(cc, axis=1)
    ee = np.sqrt(dd)
    ff = np.argwhere(ee < distance_thresh)
    del_list = []
    for i in range(ff.size):
        index = ff[i][0]
        del_list.append(index)

    endpoints_list = np.delete(endpoints_list, del_list, 0)

    # save csv file endpoint coordinates
    csv_endpoint_file = 'endpoint_' + csv_file
    np.savetxt(csv_endpoint_file, endpoints_list, delimiter=",")

    # # save endpoints in niffti format
    # centerline_name = 'endpoints_.nii.gz'
    # filename1 = os.path.join(niftidir, intersect)
    # img1 = nib.load(filename1)
    # mask = nib.Nifti1Image(endpoints, img1.affine, img1.header)
    # nib.save(mask, os.path.join(niftidir, centerline_name))

    return co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal, intersect_coordinates


def branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z, intersect_coordinate,
                     degree_angle, branch_name, neighbour_name1, neighbour_name2):

    label = STRUCT[branch_name]
    indx = np.argwhere(df_numpy[:, 3] == label)

    max_indx = indx.size - 1
    if max_indx > 50:
        max_indx = 50

    start_x_proximal = df_numpy[indx[0], 0]
    start_y_proximal = df_numpy[indx[0], 1]
    start_z_proximal = df_numpy[indx[0], 2]

    end_x_proximal = df_numpy[indx[-1], 0]
    end_y_proximal = df_numpy[indx[-1], 1]
    end_z_proximal = df_numpy[indx[-1], 2]

    diff = np.abs(start_z_proximal - end_z_proximal)
    # decision for using co_occurrence or co_occurrence_inverse matrix to calculate neighbour
    if diff <= 3:
        xx1 = np.abs(start_x_proximal - intersect_coordinate[0])
        yy1 = np.abs(start_y_proximal - intersect_coordinate[1])
        xy_start = xx1 + yy1

        xx2 = np.abs(end_x_proximal - intersect_coordinate[0])
        yy2 = np.abs(end_y_proximal - intersect_coordinate[1])
        xy_end = xx2 + yy2
        if xy_start <= xy_end:
            neighbour = np.argwhere(co_occurrence[label, :] == 1)
            previous_z = end_z_proximal

            start_x_proximal = df_numpy[indx[-max_indx], 0]
            start_y_proximal = df_numpy[indx[-max_indx], 1]

            deltaY = end_y_proximal - start_y_proximal
            deltaX = end_x_proximal - start_x_proximal

        else:
            neighbour = np.argwhere(co_occurrence_inverse[label, :] == 1)
            previous_z = start_z_proximal

            end_x_proximal = df_numpy[indx[max_indx], 0]
            end_y_proximal = df_numpy[indx[max_indx], 1]

            deltaY = start_y_proximal - end_y_proximal
            deltaX = start_x_proximal - end_x_proximal

    elif (np.abs(end_z_proximal - previous_z) > np.abs(start_z_proximal - previous_z)):
        # finding neighbour of this branch
        neighbour = np.argwhere(co_occurrence[label, :] == 1)
        previous_z = end_z_proximal

        start_x_proximal = df_numpy[indx[-max_indx], 0]
        start_y_proximal = df_numpy[indx[-max_indx], 1]

        deltaY = end_y_proximal - start_y_proximal
        deltaX = end_x_proximal - start_x_proximal

    else:
        neighbour = np.argwhere(co_occurrence_inverse[label, :] == 1)
        previous_z = start_z_proximal

        end_x_proximal = df_numpy[indx[max_indx], 0]
        end_y_proximal = df_numpy[indx[max_indx], 1]

        deltaY = start_y_proximal - end_y_proximal
        deltaX = start_x_proximal - end_x_proximal

    # calculation angle between branches and horizontal axes
    proximal_angle = atan2(deltaY, deltaX) * 180 / pi

    num_neighbour = np.shape(neighbour)[0]
    angle_matrix = np.zeros([num_neighbour, 2])

    # comparing branch angle and its neighbour angle
    for i in range(num_neighbour):
        label = neighbour[i]
        label = label[0]
        indx = np.argwhere(df_numpy[:, 3] == label)

        max_indx = indx.size - 1
        if max_indx > 50:
            max_indx = 50


        start_x_proximal = df_numpy[indx[0], 0]
        start_y_proximal = df_numpy[indx[0], 1]
        start_z_proximal = df_numpy[indx[0], 2]

        end_x_proximal = df_numpy[indx[-1], 0]
        end_y_proximal = df_numpy[indx[-1], 1]
        end_z_proximal = df_numpy[indx[-1], 2]

        if (np.abs(end_z_proximal - previous_z) >= np.abs(start_z_proximal - previous_z)):

            end_x_proximal = df_numpy[indx[max_indx], 0]
            end_y_proximal = df_numpy[indx[max_indx], 1]

            deltaY = end_y_proximal - start_y_proximal
            deltaX = end_x_proximal - start_x_proximal
        else:

            start_x_proximal = df_numpy[indx[-max_indx], 0]
            start_y_proximal = df_numpy[indx[-max_indx], 1]

            deltaY = start_y_proximal - end_y_proximal
            deltaX = start_x_proximal - end_x_proximal

        angle = atan2(deltaY, deltaX) * 180 / pi

        diff_angle = np.abs(proximal_angle - angle)

        angle_matrix[i, 0] = diff_angle
        angle_matrix[i, 1] = label

        if diff_angle <= degree_angle:
            STRUCT[neighbour_name1] = label
        else:
            STRUCT[neighbour_name2] = label

    index = np.argsort(angle_matrix[:, 0], axis=0)

    # decision for some branches that both neighbour angles are bigger or smaller than proximal angle
    if num_neighbour == 2 and (STRUCT[neighbour_name1] == -1 or STRUCT[neighbour_name2] == -1):
        STRUCT[neighbour_name1] = angle_matrix[index[0], 1].astype('int64')
        STRUCT[neighbour_name2] = angle_matrix[index[1], 1].astype('int64')

    return previous_z, STRUCT


def branch_labeling_left(niftidir, csv_file, co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal,
                         intersect_coordinates):

    filename1 = os.path.join(niftidir, csv_file)
    df_numpy = genfromtxt(filename1, delimiter=',')
    df_numpy = df_numpy.astype(np.int32)
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

    # proximal
    STRUCT['aorta'] = [labels_near_aorta + 1]
    aorta_z = intersect_coordinates[2]
    previous_z = aorta_z

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

    # branch 1
    branch_name = 'Left main'
    label = STRUCT[branch_name]
    degree_angle = 40

    thresh_add_proximal = 0
    thresh_add = 0
    thresh_length = 210000 + thresh_add

    split_add_proximal = 0
    split_add = 0
    thresh_length_split = 210000 + split_add


    if label != -1:
        neighbour_name1 = 'Proximal LAD'
        neighbour_name2 = 'Proximal LCX'
        # measure length
        direction, previous_z_temp = centerline_direction(df_numpy, label, previous_z)

        index = np.argwhere(df_numpy[:, 3] == label)

        max_index = index.size - 1
        if max_index > 50:
            max_index = 50

        cent_array = df_numpy[index, 0:3]
        if direction:
            center_length, num_split = length(cent_array, thresh_length_split)
        else:
            cent_array = np.flip(cent_array, 0)
            center_length, num_split = length(cent_array, thresh_length_split)
            cent_size = np.shape(cent_array)[0]
            num_split_inverse = cent_size - num_split - 1

        if num_split > 0:
            # split centerline

            if direction:
                start_x_proximal = df_numpy[index[0], 0]
                start_y_proximal = df_numpy[index[0], 1]

                end_x_proximal = df_numpy[index[num_split], 0]
                end_y_proximal = df_numpy[index[num_split], 1]

                deltaY = end_y_proximal - start_y_proximal
                deltaX = end_x_proximal - start_x_proximal

                proximal_angle = atan2(deltaY, deltaX) * 180 / pi

                start_x = df_numpy[index[num_split], 0]
                start_y = df_numpy[index[num_split], 1]

                end_x = df_numpy[index[num_split + max_index], 0]
                end_y = df_numpy[index[num_split + max_index], 1]

                deltaY = end_y - start_y
                deltaX = end_x - start_x

                angle = atan2(deltaY, deltaX) * 180 / pi

                diff_angle = np.abs(proximal_angle - angle)

                if diff_angle <= 51:
                    STRUCT[neighbour_name1] = label
                    branch_label_map[index[num_split::]] = 6
                    branch_parenthood[index[num_split::]] = label_proximal + 1
                else:
                    STRUCT[neighbour_name2] = label
                    branch_label_map[index[num_split::]] = 11
                    branch_parenthood[index[num_split::]] = label_proximal + 1

            else:
                start_x_proximal = df_numpy[index[-1], 0]
                start_y_proximal = df_numpy[index[-1], 1]

                end_x_proximal = df_numpy[index[num_split_inverse], 0]
                end_y_proximal = df_numpy[index[num_split_inverse], 1]

                deltaY = start_y_proximal - end_y_proximal
                deltaX = start_x_proximal - end_x_proximal

                proximal_angle = atan2(deltaY, deltaX) * 180 / pi

                start_x = df_numpy[index[num_split_inverse], 0]
                start_y = df_numpy[index[num_split_inverse], 1]

                end_x = df_numpy[index[num_split_inverse - max_index], 0]
                end_y = df_numpy[index[num_split_inverse - max_index], 1]

                deltaY = start_y - end_y
                deltaX = start_x - end_x

                angle = atan2(deltaY, deltaX) * 180 / pi

                diff_angle = np.abs(proximal_angle - angle)

                if diff_angle <= 50:
                    STRUCT[neighbour_name1] = label
                    branch_label_map[index[0:num_split_inverse]] = 6
                    branch_parenthood[index[0:num_split_inverse]] = label_proximal + 1
                else:
                    STRUCT[neighbour_name2] = label
                    branch_label_map[index[0:num_split_inverse]] = 11
                    branch_parenthood[index[0:num_split_inverse]] = label_proximal + 1

            thresh_add = thresh_add + thresh_length
            split_add = split_add + thresh_length_split

            thresh_add_proximal = thresh_length
            split_add_proximal = thresh_length_split

        else:
            previous_z, STRUCT = branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z,
                                                  intersect_coordinates, degree_angle, branch_name, neighbour_name1,
                                                  neighbour_name2)
            thresh_add = 0
            split_add = 0

            label = STRUCT[neighbour_name1]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 6
            branch_parenthood[index] = label_proximal + 1

            label = STRUCT[neighbour_name2]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 11
            branch_parenthood[index] = label_proximal + 1

    # branch 1.1
    branch_name = 'Proximal LCX'
    label = STRUCT[branch_name]
    degree_angle = 40

    thresh_length = 100000 + thresh_add
    thresh_length_split = 100000 + split_add

    if label != -1:
        neighbour_name1 = 'Mid-distal LCX'
        neighbour_name2 = 'First marginal'
        # measure length
        direction, previous_z_temp = centerline_direction(df_numpy, label, previous_z)

        index = np.argwhere(df_numpy[:, 3] == label)
        cent_array = df_numpy[index, 0:3]
        if direction:
            center_length, num_split = length(cent_array, thresh_length_split)
        else:
            cent_array = np.flip(cent_array, 0)
            center_length, num_split = length(cent_array, thresh_length_split)
            cent_size = np.shape(cent_array)[0]
            num_split_inverse = cent_size - num_split - 1

        if num_split > 0:
            # split centerline
            STRUCT[neighbour_name1] = label

            if direction:
                branch_label_map[index[num_split::]] = 13
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[num_split::]] = parenthood
            else:
                branch_label_map[index[0:num_split_inverse]] = 13
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[0:num_split_inverse]] = parenthood

            thresh_add = thresh_add + thresh_length
            split_add = split_add + thresh_length_split

        else:
            previous_z, STRUCT = branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z,
                                                  intersect_coordinates, degree_angle, branch_name, neighbour_name1,
                                                  neighbour_name2)
            thresh_add = 0
            split_add = 0

            label = STRUCT[neighbour_name1]
            parenthood = STRUCT[branch_name]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 13
            branch_parenthood[index] = parenthood

            label = STRUCT[neighbour_name2]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 12
            branch_parenthood[index] = parenthood

    # branch 1.2
    branch_name = 'Mid-distal LCX'
    label = STRUCT[branch_name]
    degree_angle = 20

    thresh_length = 439000 + thresh_add
    thresh_length_split = 439000 + split_add

    if label != -1:
        neighbour_name1 = 'Left PDA'
        neighbour_name2 = 'Posterolateral branch'
        # measure length
        direction, previous_z_temp = centerline_direction(df_numpy, label, previous_z)

        index = np.argwhere(df_numpy[:, 3] == label)
        cent_array = df_numpy[index, 0:3]
        if direction:
            center_length, num_split = length(cent_array, thresh_length_split)
        else:
            cent_array = np.flip(cent_array, 0)
            center_length, num_split = length(cent_array, thresh_length_split)
            cent_size = np.shape(cent_array)[0]
            num_split_inverse = cent_size - num_split - 1

        if num_split > 0:
            # split centerline
            STRUCT[neighbour_name1] = label

            if direction:
                branch_label_map[index[num_split::]] = 15
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[num_split::]] = parenthood
            else:
                branch_label_map[index[0:num_split_inverse]] = 15
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[0:num_split_inverse]] = parenthood

            thresh_add = thresh_add + thresh_length
            split_add = split_add + thresh_length_split

        else:
            previous_z, STRUCT = branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z,
                                                  intersect_coordinates, degree_angle, branch_name, neighbour_name1,
                                                  neighbour_name2)
            thresh_add = 0
            split_add = 0

            label = STRUCT[neighbour_name1]
            parenthood = STRUCT[branch_name]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 15
            branch_parenthood[index] = parenthood

            label = STRUCT[neighbour_name2]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 14
            branch_parenthood[index] = parenthood

    # branch 2
    branch_name = 'Proximal LAD'
    label = STRUCT[branch_name]
    degree_angle = 20
    previous_z = aorta_z

    thresh_add = thresh_add_proximal
    split_add = split_add_proximal
    thresh_length = 154000 + thresh_add
    thresh_length_split = 154000 + split_add

    if label != -1:
        neighbour_name1 = 'Mid LAD'
        neighbour_name2 = 'First Diagonal'
        # measure length
        direction, previous_z_temp = centerline_direction(df_numpy, label, previous_z)

        index = np.argwhere(df_numpy[:, 3] == label)
        cent_array = df_numpy[index, 0:3]
        if direction:
            center_length, num_split = length(cent_array, thresh_length_split)
        else:
            cent_array = np.flip(cent_array, 0)
            center_length, num_split = length(cent_array, thresh_length_split)
            cent_size = np.shape(cent_array)[0]
            num_split_inverse = cent_size - num_split - 1

        if num_split > 0:
            # split centerline
            STRUCT[neighbour_name1] = label

            if direction:
                branch_label_map[index[num_split::]] = 7
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[num_split::]] = parenthood
            else:
                branch_label_map[index[0:num_split_inverse]] = 7
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[0:num_split_inverse]] = parenthood

            thresh_add = thresh_add + thresh_length
            split_add = split_add + thresh_length_split

        else:
            previous_z, STRUCT = branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z,
                                                  intersect_coordinates, degree_angle, branch_name, neighbour_name1,
                                                  neighbour_name2)
            thresh_add = 0
            split_add = 0

            label = STRUCT[neighbour_name1]
            parenthood = STRUCT[branch_name]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 7
            branch_parenthood[index] = parenthood

            label = STRUCT[neighbour_name2]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 9
            branch_parenthood[index] = parenthood

    # branch 2.1
    branch_name = 'Mid LAD'
    label = STRUCT[branch_name]
    degree_angle = 40

    thresh_length = 115000 + thresh_add
    thresh_length_split = 115000 + split_add

    if label != -1:
        neighbour_name1 = 'Distal LAD'
        neighbour_name2 = 'Second Diagonal'
        # measure length
        direction, previous_z_temp = centerline_direction(df_numpy, label, previous_z)

        index = np.argwhere(df_numpy[:, 3] == label)
        cent_array = df_numpy[index, 0:3]
        if direction:
            center_length, num_split = length(cent_array, thresh_length_split)
        else:
            cent_array = np.flip(cent_array, 0)
            center_length, num_split = length(cent_array, thresh_length_split)
            cent_size = np.shape(cent_array)[0]
            num_split_inverse = cent_size - num_split - 1

        if num_split > 0:
            # split centerline
            STRUCT[neighbour_name1] = label

            if direction:
                branch_label_map[index[num_split::]] = 8
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[num_split::]] = parenthood
            else:
                branch_label_map[index[0:num_split_inverse]] = 8
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[0:num_split_inverse]] = parenthood

            thresh_add = thresh_add + thresh_length
            split_add = split_add + thresh_length_split

        else:
            previous_z, STRUCT = branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z,
                                                  intersect_coordinates, degree_angle, branch_name, neighbour_name1,
                                                  neighbour_name2)
            thresh_add = 0
            split_add = 0

            label = STRUCT[neighbour_name1]
            parenthood = STRUCT[branch_name]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 8
            branch_parenthood[index] = parenthood

            label = STRUCT[neighbour_name2]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 10
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

    json_string = '['
    branch_name = 'Left main'
    if STRUCT[branch_name] != -1:
        parenthood = 0
        label_map = 5
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Proximal LAD'
    if STRUCT[branch_name] != -1:
        parenthood = 5
        label_map = 6
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Mid LAD'
    if STRUCT[branch_name] != -1:
        parenthood = 6
        label_map = 7
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Distal LAD'
    if STRUCT[branch_name] != -1:
        parenthood = 7
        label_map = 8
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'First Diagonal'
    if STRUCT[branch_name] != -1:
        parenthood = 6
        label_map = 9
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Second Diagonal'
    if STRUCT[branch_name] != -1:
        parenthood = 7
        label_map = 10
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Proximal LCX'
    if STRUCT[branch_name] != -1:
        parenthood = 5
        label_map = 11
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Mid-distal LCX'
    if STRUCT[branch_name] != -1:
        parenthood = 11
        label_map = 13
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'First marginal'
    if STRUCT[branch_name] != -1:
        parenthood = 11
        label_map = 12
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Posterolateral branch'
    if STRUCT[branch_name] != -1:
        parenthood = 13
        label_map = 14
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Left PDA'
    if STRUCT[branch_name] != -1:
        parenthood = 13
        label_map = 15
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    # for branches without label
    branch_name = 'unknown'
    parenthood = -1
    label_map = -1
    json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
    json_string = json_string + json_string1

    json_string = json_string + ']'

    os.chdir(niftidir)
    filename = csv_file + '.txt'

    text_file = open(filename, "wt")
    n = text_file.write(json_string)
    text_file.close()

    # with open(filename, 'w') as outfile:
    #     json.dump(json_string, outfile)


def length(cent_array, split_dist):

    cent_size = np.shape(cent_array)[0]
    cent_array1 = np.reshape(cent_array, [cent_size, 3])
    fist_array = cent_array1[0, :]
    end_array = cent_array1[-1, :]

    cent_array2 = cent_array1

    cent_array1 = np.insert(cent_array1, 0, fist_array)
    cent_array1 = np.reshape(cent_array1, [cent_size + 1, 3])

    cent_array2 = np.append(cent_array2, end_array)
    cent_array2 = np.reshape(cent_array2, [cent_size + 1, 3])

    diff = np.sum(np.abs(cent_array1, cent_array2))

    step = int(np.round(cent_size/10) - 2)
    num_split = 0
    for i in range(step):
        indx = i * 10
        diff_part = np.sum(np.abs(cent_array1[0:indx, :], cent_array2[0:indx, :]))
        if diff_part > split_dist:
            num_split = indx
            break

    return diff, num_split


def centerline_direction(df_numpy, label, previous_z):
    indx = np.argwhere(df_numpy[:, 3] == label)

    start_z_proximal = df_numpy[indx[0], 2]

    end_z_proximal = df_numpy[indx[-1], 2]

    if (np.abs(end_z_proximal - previous_z) >= np.abs(start_z_proximal - previous_z)):
        direction = True
        previous_z = end_z_proximal
    else:
        direction = False
        previous_z = start_z_proximal

    return direction, previous_z


def branch_labeling_right(niftidir, csv_file, co_occurrence, co_occurrence_inverse, labels_near_aorta, label_proximal,
                          intersect_coordinates):

    filename1 = os.path.join(niftidir, csv_file)
    df_numpy = genfromtxt(filename1, delimiter=',')
    df_numpy = df_numpy.astype(np.int32)
    branch_label_map = np.ones([np.shape(df_numpy)[0], 1]) * -1
    branch_parenthood = np.ones([np.shape(df_numpy)[0], 1]) * -1

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

    # proximal
    STRUCT['aorta'] = [labels_near_aorta + 1]
    aorta_z = intersect_coordinates[2]
    previous_z = aorta_z

    # set label 1 to all labels left main
    num_labels = np.shape(labels_near_aorta)[0]
    for i in range(num_labels):
        label = STRUCT['aorta'][0][i]
        index = np.argwhere(df_numpy[:, 3] == label)
        branch_label_map[index] = 0
        branch_parenthood[index] = 0

    label = label_proximal + 1
    STRUCT['Proximal RCA'] = label
    index = np.argwhere(df_numpy[:, 3] == label)
    branch_label_map[index] = 1
    branch_parenthood[index] = 0

    # branch 1
    branch_name = 'Proximal RCA'
    label = STRUCT[branch_name]
    degree_angle = 40

    thresh_add = 0
    thresh_length = 231000 + thresh_add

    split_add = 0
    thresh_length_split = 231000 + split_add

    if label != -1:
        neighbour_name1 = 'V'
        neighbour_name2 = 'Mid RCA'

        # measure length
        direction, previous_z_temp = centerline_direction(df_numpy, label, previous_z)

        index = np.argwhere(df_numpy[:, 3] == label)
        cent_array = df_numpy[index, 0:3]
        if direction:
            center_length, num_split = length(cent_array, thresh_length_split)
        else:
            cent_array = np.flip(cent_array, 0)
            center_length, num_split = length(cent_array, thresh_length_split)
            cent_size = np.shape(cent_array)[0]
            num_split_inverse = cent_size - num_split - 1

        if num_split > 0:
            # split centerline
            STRUCT[neighbour_name2] = label

            if direction:
                branch_label_map[index[num_split::]] = 2
                branch_parenthood[index[num_split::]] = label_proximal + 1
            else:
                branch_label_map[index[0:num_split_inverse]] = 2
                branch_parenthood[index[0:num_split_inverse]] = label_proximal + 1

            thresh_add = thresh_add + thresh_length
            split_add = split_add + thresh_length_split

        else:
            previous_z, STRUCT = branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z,
                                                  intersect_coordinates, degree_angle, branch_name, neighbour_name1,
                                                  neighbour_name2)
            thresh_add = 0
            split_add = 0

            label = STRUCT[neighbour_name1]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 16
            branch_parenthood[index] = label_proximal + 1

            label = STRUCT[neighbour_name2]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 2
            branch_parenthood[index] = label_proximal + 1

    # branch 1.1
    branch_name = 'Mid RCA'
    label = STRUCT[branch_name]
    degree_angle = 40

    thresh_length = 200000 + thresh_add
    thresh_length_split = 200000 + split_add

    if label != -1:
        neighbour_name1 = 'Distal RCA'
        neighbour_name2 = 'AM'

        # measure length
        direction, previous_z_temp = centerline_direction(df_numpy, label, previous_z)

        index = np.argwhere(df_numpy[:, 3] == label)
        cent_array = df_numpy[index, 0:3]
        if direction:
            center_length, num_split = length(cent_array, thresh_length_split)
        else:
            cent_array = np.flip(cent_array, 0)
            center_length, num_split = length(cent_array, thresh_length_split)
            cent_size = np.shape(cent_array)[0]
            num_split_inverse = cent_size - num_split - 1

        if num_split > 0:
            # split centerline
            STRUCT[neighbour_name1] = label

            if direction:
                branch_label_map[index[num_split::]] = 3
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[num_split::]] = parenthood
            else:
                branch_label_map[index[0:num_split_inverse]] = 3
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[0:num_split_inverse]] = parenthood

            thresh_add = thresh_add + thresh_length
            split_add = split_add + thresh_length_split

        else:
            previous_z, STRUCT = branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z,
                                                  intersect_coordinates, degree_angle, branch_name, neighbour_name1,
                                                  neighbour_name2)
            thresh_add = 0
            split_add = 0

            label = STRUCT[neighbour_name1]
            parenthood = STRUCT[branch_name]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 3
            branch_parenthood[index] = parenthood

            label = STRUCT[neighbour_name2]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 17
            branch_parenthood[index] = parenthood

    # branch 1.2
    branch_name = 'Distal RCA'
    label = STRUCT[branch_name]
    degree_angle = 40

    thresh_length = 390000 + thresh_add
    thresh_length_split = 390000 + split_add

    if label != -1:
        neighbour_name1 = 'RPD'
        neighbour_name2 = 'Right PDA'

        # measure length
        direction, previous_z_temp = centerline_direction(df_numpy, label, previous_z)

        index = np.argwhere(df_numpy[:, 3] == label)
        cent_array = df_numpy[index, 0:3]
        if direction:
            center_length, num_split = length(cent_array, thresh_length_split)
        else:
            cent_array = np.flip(cent_array, 0)
            center_length, num_split = length(cent_array, thresh_length_split)
            cent_size = np.shape(cent_array)[0]
            num_split_inverse = cent_size - num_split - 1

        if num_split > 0:
            # split centerline
            STRUCT[neighbour_name2] = label

            if direction:
                branch_label_map[index[num_split::]] = 4
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[num_split::]] = parenthood
            else:
                branch_label_map[index[0:num_split_inverse]] = 4
                parenthood = STRUCT[branch_name]
                branch_parenthood[index[0:num_split_inverse]] = parenthood
            thresh_add = thresh_add + thresh_length
            split_add = split_add + thresh_length_split

        else:
            previous_z, STRUCT = branch_neighbour(df_numpy, co_occurrence, co_occurrence_inverse, STRUCT, previous_z,
                                                  intersect_coordinates, degree_angle, branch_name, neighbour_name1,
                                                  neighbour_name2)
            thresh_add = 0
            split_add = 0

            label = STRUCT[neighbour_name1]
            parenthood = STRUCT[branch_name]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 18
            branch_parenthood[index] = parenthood

            label = STRUCT[neighbour_name2]
            index = np.argwhere(df_numpy[:, 3] == label)
            branch_label_map[index] = 4
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

    json_string = '['
    branch_name = 'Proximal RCA'
    if STRUCT[branch_name] != -1:
        parenthood = 0
        label_map = 1
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Mid RCA'
    if STRUCT[branch_name] != -1:
        parenthood = 1
        label_map = 2
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Distal RCA'
    if STRUCT[branch_name] != -1:
        parenthood = 2
        label_map = 3
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'Right PDA'
    if STRUCT[branch_name] != -1:
        parenthood = 3
        label_map = 4
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'V'
    if STRUCT[branch_name] != -1:
        parenthood = 1
        label_map = 16
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'AM'
    if STRUCT[branch_name] != -1:
        parenthood = 2
        label_map = 17
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    branch_name = 'RPD'
    if STRUCT[branch_name] != -1:
        parenthood = 3
        label_map = 18
        json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
        json_string = json_string + json_string1
        json_string = json_string + ','

    # for branches without label
    branch_name = 'unknown'
    parenthood = -1
    label_map = -1
    json_string1 = dict_to_json(branch_name, parenthood, label_map, output)
    json_string = json_string + json_string1

    json_string = json_string + ']'

    os.chdir(niftidir)
    filename = csv_file + '.txt'

    text_file = open(filename, "wt")
    n = text_file.write(json_string)
    text_file.close()

    # with open(filename, 'w') as outfile:
    #     json.dump(json_string, outfile)


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


def cmpr_preparation(niftidir, niftiroot, dilated_cor_aorta_name, csv_endpoint_file, cx, cy, cz):
    from vmtk import pypes
    from vmtk import vmtkscripts

    os.chdir(niftidir)

    # automatically centerline calculation

    surface = niftiroot + '_cor_aorta.vtp'
    myargs = 'vmtkmarchingcubes -ifile ' + dilated_cor_aorta_name + ' -l 1.0 -ofile ' + surface
    mypype = pypes.PypeRun(myargs)

    smoothing_iteration = 2700
    surface_smooth = niftiroot + '_smooth.vtp'
    myargs = 'vmtksurfacesmoothing -iterations ' + str(
        smoothing_iteration) + ' -ifile ' + surface + ' -ofile ' + surface_smooth
    mypype = pypes.PypeRun(myargs)

    filename1 = os.path.join(niftidir, csv_endpoint_file)
    df_numpy = genfromtxt(filename1, delimiter=',')
    df_numpy = df_numpy.astype(np.int32)

    if df_numpy.shape.__len__() == 1:
        df_numpy_size = 1
    else:
        df_numpy_size = df_numpy.shape[0]

    for i in range(df_numpy_size):

        if df_numpy_size == 1:
            x = df_numpy[0]
            y = df_numpy[1]
            z = df_numpy[2]
            label = df_numpy[3]
        else:
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

        # neighbour point
        targetpointId_surf_neighbour = str(Surface.FindPoint(targetpoint_img) + 1)
        space_str = ' '
        targetpointId_surf = targetpointId_surf + space_str + targetpointId_surf_neighbour

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
