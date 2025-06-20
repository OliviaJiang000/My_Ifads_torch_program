import numpy as np
from scipy.io import loadmat

# 读取 .mat 文件
data = loadmat("/Users/jojo/my_ifads_experiment/datasets/dataset001.mat")
spikes = data["spikes"]  # [trials, time, neurons]

# 一些参数
dt = 0.01
n_trials = spikes.shape[0]
train_mask = np.zeros(n_trials, dtype=bool)
train_mask[:int(n_trials * 0.8)] = True
valid_mask = ~train_mask

# 保存为 .npz
np.savez("myData/lfads_ready_dataset.npz", spikes=spikes, dt=dt,
         train_mask=train_mask, valid_mask=valid_mask)
