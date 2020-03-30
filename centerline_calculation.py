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


    myargs = 'vmtksurfacereader -ifile ' + surface_clip + ' --pipe ' + \
             'vmtkrenderer --pipe ' + \
             'vmtksurfaceviewer -opacity 0.25' + ' --pipe ' + \
             'vmtksurfaceviewer -ifile ' + surface_centerline + ' -array MaximumInscribedSphereRadius'

    mypype = pypes.PypeRun(myargs)


    # export senterline coordinates

    centerlineReader = vmtkscripts.vmtkSurfaceReader()
    centerlineReader.InputFileName = surface_centerline
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


def csv_to_nifti(niftidir, size_data, csv_file):

    import pandas as pd

    centerline = np.zeros(size_data)

    filename1 = os.path.join(niftidir, csv_file)
    df = pd.read_csv(filename1)
    print(df)

    for i in range(df.values.shape[0]):
        x = round(df.values[i, 0]).astype(np.int32)
        y = round(df.values[i, 1]).astype(np.int32)
        z = round(df.values[i, 2]).astype(np.int32)
        label = round(df.values[i, 3]).astype(np.int32)
        centerline[x, y, z] = label

    kernel = ball(3)
    dilated_centerline = centerline
    for i in range(label):
        mask = centerline == (i + 1)
        mask = binary_dilation(mask, kernel)
        dilated_centerline = np.where(mask > 0, i + 1, dilated_centerline)

    points = dilated_centerline

    return points


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
    mask = nib.Nifti1Image(intersect, img1.affine, img1.header)

    intersect_L = niftiroot_L +'_intersect.nii.gz'
    nib.save(mask, os.path.join(niftidir, intersect_L))

    # Right side

    intersect = dilated_right_cor * dilated_aorta
    mask = nib.Nifti1Image(intersect, img1.affine, img1.header)

    intersect_R = niftiroot_R +'_intersect.nii.gz'
    nib.save(mask, os.path.join(niftidir, intersect_R))

    # calculation left side centerline
    csv_file_L = centerline_calculation(niftidir, niftiname_L, niftiroot_L, intersect_L)

    # calculation right side centerline
    csv_file_R = centerline_calculation(niftidir, niftiname_R, niftiroot_R, intersect_R)


    points = csv_to_nifti(niftidir, np.shape(coronary), csv_file_L)
    centerline_name = 'centreline_' + niftiroot_L + '.nii.gz'
    mask = nib.Nifti1Image(points, img1.affine, img1.header)
    nib.save(mask, os.path.join(niftidir, centerline_name))

    points = csv_to_nifti(niftidir, np.shape(coronary), csv_file_R)
    centerline_name = 'centreline_' + niftiroot_R + '.nii.gz'
    mask = nib.Nifti1Image(points, img1.affine, img1.header)
    nib.save(mask, os.path.join(niftidir, centerline_name))




if __name__ == '__main__':
    main()


