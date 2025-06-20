import h5py
import numpy as np

# 修改为你自己的路径
file_path = './scripts/runs/lfads-torch-example/nlb_mc_maze/250612_exampleSingle/lfads_output_mc_maze-20ms-val.h5'

file_path='/Users/jojo/Documents/PythonProject/My_IFads_torch_program/lfads-torch/datasets/mc_maze_small-05ms-val.h5'
with h5py.File(file_path, 'r') as f:
    print(list(f.keys()))  # 查看所有内容


    def print_structure(name, obj):
        print(name)


    f.visititems(print_structure)
    # 示例读取
    train_factors = f['train_factors'][:]       # shape: (trials, time, dim)
    valid_factors = f['valid_factors'][:]
    train_recon = f['train_recon_data'][:]      # 重建出来的 spike count（firing rates）
    valid_recon = f['valid_recon_data'][:]

    print(train_factors.shape)
    print(valid_factors.shape)


# . 单个 trial 的 factor 三维轨迹：
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

trial_idx = 0
f1, f2, f3 = valid_factors[trial_idx, :, :3].T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(f1, f2, f3)
ax.set_title(f'LFADS Factor Trajectory - Trial {trial_idx}')
plt.show()


# 2. 某个神经元的 rate 重建 vs time：
# neuron_id = 0
# plt.plot(rates[trial_idx, :, neuron_id])
# plt.title(f'Reconstructed Firing Rate - Neuron {neuron_id}')
# plt.xlabel("Time")
# plt.ylabel("Firing Rate")
# plt.show()

# 第三步：评估模型质量（如和
# ground
# truth
# 对比）

# original = f['truth'][:]  # (trials, time, neurons)
#
# from sklearn.metrics import r2_score
# r2 = r2_score(original.flatten(), rates.flatten())
# print("R² for reconstruction:", r2)
