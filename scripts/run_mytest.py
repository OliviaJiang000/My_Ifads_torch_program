from nlb_tools.nwb_interface import NWBDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ## Load data
# dataset_name = 'mc_maze_large'
# datapath = './000138/sub-Jenkins/'
# dataset = NWBDataset(datapath)
# # dataset = NWBDataset("~/lvm/code/dandi/000127/sub-Han/", "*train", split_heldout=False)
#
#
# print(dataset)


from pynwb import NWBHDF5IO

file_path = '/Users/jojo/Documents/PythonProject/My_IFads_torch_program/lfads-torch/000127/sub-Han/sub-Han_desc-test_ecephys.nwb'
dataset = NWBDataset(file_path)
print(dataset)
# with NWBHDF5IO(file_path, 'r') as io:
#     nwbfile = io.read()
#
#     # 提取所有神经元 ID 和 spike 时间
#     units = nwbfile.units
#     unit_ids = units.id[:]
#     spike_times = units['spike_times'][:]
#
#     # 查看 behavioral_events 或 trials
#     if nwbfile.trials is not None:
#         print(nwbfile.trials.colnames)
#         print(nwbfile.trials['start_time'][:5])

