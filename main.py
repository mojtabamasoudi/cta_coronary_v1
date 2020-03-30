# IMPORTS --------------------------------------------------------------------------------------------------------------

import sys
import os
from vmtk import pypes
from vmtk import vmtkscripts
import numpy as np
import nibabel as nib
import pandas as pd
from skimage.morphology import ball, binary_dilation
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

niftidir = r'D:\Mojtaba\Dataset_test\9'
os.chdir(niftidir)
niftiname = 'R_coronary.nii.gz'
niftiroot = 'R_coronary'

initial_surface = 'aorta_R_intersect.nii.gz'

# automatically centerline calculation
myargs = 'vmtkmarchingcubes -ifile ' + niftiname + ' -l 1.0 -ofile vessel_aorta_surface.vtp ' \
          '--pipe vmtksurfaceviewer -array GroupIds'
mypype = pypes.PypeRun(myargs)


myargs = 'vmtksurfacesmoothing -iterations 3000 -ifile vessel_aorta_surface.vtp -ofile vessel_aorta_surface_smooth.vtp'
mypype = pypes.PypeRun(myargs)


myargs = 'vmtkmarchingcubes -ifile ' + initial_surface + ' -l 1.0 -ofile aorta_L_intersect_surface.vtp'
mypype = pypes.PypeRun(myargs)


# myargs = 'vmtksurfaceclipper -ifile initial_surface.vtp -ofile OutputFile1.vtp --pipe vmtksurfaceclipper' \
#          ' -transform @.otransform -ifile vessel_aorta_surface.vtp -ofile OutputFile2.vtp'
# mypype = pypes.PypeRun(myargs)

myargs = 'vmtksurfacecliploop -ifile vessel_aorta_surface_smooth.vtp' \
         ' -i2file aorta_L_intersect_surface.vtp -ofile vessel_aorta_surface_clip.vtp'
mypype = pypes.PypeRun(myargs)


myargs = 'vmtksurfaceviewer -ifile vessel_aorta_surface_clip.vtp '
mypype = pypes.PypeRun(myargs)

myargs = 'vmtksurfaceviewer -ifile vessel_aorta_surface_clip.vtp '
mypype = pypes.PypeRun(myargs)


myargs = 'vmtknetworkextraction -ifile vessel_aorta_surface_clip.vtp -ofile vessel_aorta_centerline.vtp '
          # '-ographfile output_graph.vtp '
mypype = pypes.PypeRun(myargs)

# myargs = 'vmtksurfaceviewer -ifile output_graph.vtp '
# mypype = pypes.PypeRun(myargs)

myargs = 'vmtksurfaceviewer -ifile vessel_aorta_centerline.vtp '
mypype = pypes.PypeRun(myargs)

myargs = 'vmtksurfacereader -ifile vessel_aorta_surface_clip.vtp --pipe ' + \
         'vmtkrenderer --pipe ' + \
         'vmtksurfaceviewer -opacity 0.25' + ' --pipe ' + \
         'vmtksurfaceviewer -ifile vessel_aorta_centerline.vtp -array MaximumInscribedSphereRadius'

mypype = pypes.PypeRun(myargs)


# ----------------------------------------------------------------------------------------------------------------------
# EXPORT CENTRELINE COORDINATES
# ----------------------------------------------------------------------------------------------------------------------
centreline_file = 'vessel_aorta_centerline.vtp'


centerlineReader = vmtkscripts.vmtkSurfaceReader()
centerlineReader.InputFileName = centreline_file
centerlineReader.Execute()

clNumpyAdaptor = vmtkscripts.vmtkCenterlinesToNumpy()
clNumpyAdaptor.Centerlines = centerlineReader.Surface
clNumpyAdaptor.Execute()

numpyCenterlines = clNumpyAdaptor.ArrayDict
points = numpyCenterlines['Points']

labels = numpyCenterlines['CellData']
cell_points = labels['CellPointIds']
topology = labels['Topology']

for i in range(np.shape(cell_points)[0]):
    number = np.shape(cell_points[i])[0]
    index = np.ones(number) * (i+1)
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

points_ijk = np.reshape(points_ijk, [x+1, 4])
points_ijk = np.round(points_ijk[:, 0:4])
print(points_ijk)

csv_file = 'centreline_' + niftiroot + '.csv'
np.savetxt(csv_file, points_ijk, delimiter=",")


# centerline
filename1 = os.path.join(niftidir, niftiname)
img1 = nib.load(filename1)
data1 = img1.get_fdata()

filename2 = os.path.join(niftidir, csv_file)
df = pd.read_csv(filename2)
print(df)

centerline = np.zeros(np.shape(data1))

for i in range(df.values.shape[0]):
    x = round(df.values[i, 0]).astype(np.int32)
    y = round(df.values[i, 1]).astype(np.int32)
    z = round(df.values[i, 2]).astype(np.int32)
    label = round(df.values[i, 3]).astype(np.int32)
    centerline[x, y, z] = label

kernel = ball(3)
dilated_centerline = centerline
for i in range(label):
# for i in range(14):
    mask = centerline == (i+1)
    mask = binary_dilation(mask, kernel)
    dilated_centerline = np.where(mask > 0, i+1, dilated_centerline)

final = dilated_centerline



centerline_name = 'centreline_' + niftiroot + '.nii.gz'
mask = nib.Nifti1Image(final, img1.affine, img1.header)
nib.save(mask, os.path.join(niftidir, centerline_name))



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
