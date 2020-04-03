import sys
import os
import numpy as np
import nibabel as nib
from skimage.morphology import ball, binary_dilation


def centerline_calculation(niftidir, niftiname, niftiroot, niftiname_intersect):

    from vmtk import pypes
    from vmtk import vmtkscripts

    os.chdir(niftidir)

    # automatically centerline calculation

    surface = niftiroot + '.vtp'
    myargs = 'vmtkmarchingcubes -ifile ' + niftiname + ' -l 1.0 -ofile ' + surface
    mypype = pypes.PypeRun(myargs)


    surface_smooth = surface + '_smooth.vtp'
    myargs = 'vmtksurfacesmoothing -iterations 3000 -ifile ' + surface + ' -ofile ' + surface_smooth
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
    myargs = 'vmtklineresampling -ifile ' + surface_centerline + ' -ofile ' + surface_centerline_sampling
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


def endpoint_calculation(niftidir, size_data, csv_file):

    import pandas as pd

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

    # endpoint candidates
    index_endp_candidate = []
    for i in range(label):
        index = np.argwhere(df_numpy[:, 3] == i + 1)
        index_endp_candidate = np.append(index_endp_candidate, index[0], axis=0)
        index_endp_candidate = np.append(index_endp_candidate, index[-1], axis=0)

    labels = np.arange(label)
    labels = labels + 1

    # endpoints estimation
    for i in range(np.shape(index_endp_candidate)[0]):

        index = index_endp_candidate[i].astype(np.int32)

        x = df_numpy[index, 0]
        y = df_numpy[index, 1]
        z = df_numpy[index, 2]
        label = df_numpy[index, 3]

        patch = centerlines[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        other_label = np.delete(labels, label - 1)

        endpoint_or_not = np.isin(patch, other_label)
        if True in endpoint_or_not:
            pass
        else:
            endpoints[x, y, z] = 1

    return endpoints



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

    # generating intersection mask between right and left
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


    endpoints = endpoint_calculation(niftidir, np.shape(coronary), csv_file_L)
    centerline_name = 'endpoints_' + niftiroot_L + '.nii.gz'
    mask = nib.Nifti1Image(endpoints, img1.affine, img1.header)
    nib.save(mask, os.path.join(niftidir, centerline_name))

    endpoints = endpoint_calculation(niftidir, np.shape(coronary), csv_file_R)
    centerline_name = 'endpoints_' + niftiroot_R + '.nii.gz'
    mask = nib.Nifti1Image(endpoints, img1.affine, img1.header)
    nib.save(mask, os.path.join(niftidir, centerline_name))




if __name__ == '__main__':
    main()


