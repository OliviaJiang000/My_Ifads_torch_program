import pynwb
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize
from torch.fx.passes.graph_manipulation import size_bytes

# 读取 NWB 文件
filename = './myData/sub-Indy_desc-train_behavior+ecephys.nwb'
io = pynwb.NWBHDF5IO(filename, 'r')
nwbfile = io.read()

# 直接看 trial 数量
n_trials = len(nwbfile.trials)
print(f"总共有 {n_trials} 个 trial")
print(nwbfile.trials[-1])

# 查看有哪些单元 (单位)
units = nwbfile.units
print("Total units:", units['id'].data.shape[0])

# 获取所有神经元的 ID
unit_ids = units['id'].data[:]
print("Unit IDs:", unit_ids)

# 取出某一个神经元的 spike times (以秒为单位)
unit_index = 0  # 取第一个神经元
spike_times = units[unit_index]['spike_times']
print( units[0].columns) #['heldout', 'spike_times', 'obs_intervals', 'electrodes']
print( units['electrodes'].data[:])
# plt.plot(np.arange(0,len(units[1]['obs_intervals'].values[0])),units[1]['obs_intervals'].values[0])
# plt.show()
# plt.close()
# 事件发生的时间  raster plot
spike_times = units[5]['spike_times'].values[0]
plt.figure(figsize=(15, 3))
plt.eventplot(spike_times, orientation='horizontal', colors='black')
plt.title("Spike Raster for Unit 1")
plt.xlabel("Time (s)")
plt.show()
# print(f"Unit {unit_ids[unit_index]} spike times (first 10):", spike_times)

# movement
behavior=nwbfile.processing['behavior']
finger_pos=behavior.data_interfaces['finger_pos']
print( finger_pos)
positions=finger_pos.data[:]
print( positions)
# 绘制运动轨迹
# 2D
# plt.plot(positions[:, 0], positions[:, 1])
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.title("Hand movement trajectory")
# plt.show()

# 3D
from mpl_toolkits.mplot3d import Axes3D
n_samples = positions.shape[0]  # 649100
rate = finger_pos.rate  # 1000 Hz
starting_time = finger_pos.starting_time  # 0.0

timestamps = np.arange(n_samples) / rate + starting_time

positions = finger_pos.data[:]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 按时间颜色渐变
sc = ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                c=timestamps, cmap='viridis', s=1)

plt.colorbar(sc, label='Time (s)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.title("3D hand movement with time color")
plt.show(block=True)



df=nwbfile.units.to_dataframe()
print(df)
