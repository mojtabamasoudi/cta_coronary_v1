from vmtk import pypes
from vmtk import vmtkscripts
import os
import numpy as np
import pandas as pd




niftidir = r'D:\Mojtaba\Dataset_test\test\002_bi_shuying_ct1668532'
niftiroot = 'L_coronary'

os.chdir(niftidir)
csv_file_L = 'centreline_' + niftiroot + '.csv'

csv_file = 'branch_' + csv_file_L
filename1 = os.path.join(niftidir, csv_file)
df = pd.read_csv(filename1)
df_numpy = np.round(df.to_numpy().astype(np.int32))

import json
branch_label_map = df_numpy[:, 4]
index = np.argwhere(branch_label_map == 0)
coordinate_dict = list()
for i in range(np.shape(index)[0]):
    dict_temp = dict(X=int(df_numpy[index[i], 0][0]), Y=int(df_numpy[index[i], 1][0]), Z=int(df_numpy[index[i], 2][0]))
    coordinate_dict.append(dict_temp)

coordinate_dict = tuple(coordinate_dict)
label = 2
parenthood = 1

dict_object1 = dict(branch_name='aorta', parenthood=parenthood, label=label, coordinates=coordinate_dict)
dict_object2 = dict(branch_name='mojtaba', parenthood=21, label=label, coordinates=coordinate_dict)
json_string1 = json.dumps(dict_object1)
print(json_string1)

json_string2 = json.dumps(dict_object2)

json_string = json_string1 + json_string2
print(json_string)

with open('data.txt', 'w') as outfile:
    json.dump(json_string, outfile)


a=1