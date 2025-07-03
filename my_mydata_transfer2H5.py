import numpy as np
import h5py
import pynwb
import yaml
import os

# 参数配置
nwb_path = 'myData/sub-Indy_desc-train_behavior+ecephys.nwb'
output_h5 = 'myData/my_lfads_input.h5'
output_yaml = 'configs/datamodule/my_datamodule01.yaml'
bin_size_ms = 20  # 20ms bin

# 读取 NWB 文件
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwbfile = io.read()

units = nwbfile.units
n_units = len(units['id'].data[:])
heldout_mask = units['heldout'].data[:]

# Trial 信息
trials_table = nwbfile.trials.to_dataframe().reset_index()
start_times = trials_table['start_time'].values
end_times = trials_table['stop_time'].values
n_trials = len(start_times)

print(nwbfile.processing.keys())

# 进入 behavior module 内部看看：
behavior_module = nwbfile.processing['behavior']
print(behavior_module.data_interfaces.keys())

# Behavior: 例如提取手的位置 target_pos
# 你可以根据你的具体 NWB 结构修改这里
behavior_ts = nwbfile.processing['behavior']['finger_pos']
if behavior_ts.timestamps is not None:
    behavior_timestamps = behavior_ts.timestamps[:]
else:
    n_samples = behavior_ts.data.shape[0]
    behavior_timestamps = np.arange(n_samples) / behavior_ts.rate + behavior_ts.starting_time

behavior_data = behavior_ts.data[:]  # shape (time, 3) -- (x,y,z)

# 计算整体 bin 数
session_end_time = end_times.max()
bin_size_sec = bin_size_ms / 1000
n_total_bins = int(np.ceil(session_end_time / bin_size_sec))

# 整体 binning for spike
spike_counts = np.zeros((n_units, n_total_bins), dtype=np.int16)

for i in range(n_units):
    spikes = units['spike_times'][i]
    if spikes.size == 0:
        continue
    bin_idx = (spikes / bin_size_sec).astype(int)
    for idx in np.atleast_1d(bin_idx):
        if idx < n_total_bins:
            spike_counts[i, idx] += 1

# Behavior 数据做同步 binning
# 先做时间插值对齐 spike bins
bin_times = np.arange(n_total_bins) * bin_size_sec
behavior_interp = np.zeros((n_total_bins, behavior_data.shape[1]))
for d in range(behavior_data.shape[1]):
    behavior_interp[:, d] = np.interp(
        bin_times, behavior_timestamps, behavior_data[:, d]
    )

# 拆分 heldin / heldout
heldin_units = np.where(heldout_mask == False)[0]
heldout_units = np.where(heldout_mask == True)[0]

# 切分 trial
trial_slices = []
for start, end in zip(start_times, end_times):
    start_bin = int(np.floor(start / bin_size_sec))
    end_bin = int(np.ceil(end / bin_size_sec))
    trial_slices.append((start_bin, end_bin))

# 统一 trial 长度
max_trial_len = max(end - start for start, end in zip(start_times, end_times))
max_trial_bins = int(np.ceil(max_trial_len / bin_size_sec))

def slice_trials(unit_indices, behavior=False):
    data = []
    for start_bin, end_bin in trial_slices:
        if behavior:
            trial_bins = behavior_interp[start_bin:end_bin, :]
            pad_width = max_trial_bins - trial_bins.shape[0]
            if pad_width > 0:
                trial_bins = np.pad(trial_bins, ((0,pad_width), (0,0)))
        else:
            trial_bins = spike_counts[unit_indices, start_bin:end_bin]
            pad_width = max_trial_bins - trial_bins.shape[1]
            if pad_width > 0:
                trial_bins = np.pad(trial_bins, ((0,0), (0,pad_width)))
            trial_bins = trial_bins.T  # transpose: time, unit
        data.append(trial_bins)
    return np.array(data)

train_data = slice_trials(heldin_units)
valid_data = slice_trials(heldout_units)
train_behavior = slice_trials(None, behavior=True)
valid_behavior = train_behavior.copy()  # 目前假设 behavior 在 heldin/heldout 无区分

print("Train data shape:", train_data.shape)
print("Valid data shape:", valid_data.shape)
print("Behavior data shape:", train_behavior.shape)

# 保存 h5
with h5py.File(output_h5, 'w') as f:
    f.create_dataset('train_encod_data', data=train_data.astype(np.float32))
    f.create_dataset('train_recon_data', data=train_data.astype(np.float32))
    f.create_dataset('valid_encod_data', data=valid_data.astype(np.float32))
    f.create_dataset('valid_recon_data', data=valid_data.astype(np.float32))
    f.create_dataset('train_behavior', data=train_behavior.astype(np.float32))
    f.create_dataset('valid_behavior', data=valid_behavior.astype(np.float32))

io.close()
print(f"Saved to {output_h5}")

# 自动生成新版 datamodule.yaml
seq_len = train_data.shape[1]
datamodule_config = {

        '_target_': 'lfads_torch.datamodules.BasicDataModule',
        'datafile_pattern': '${relpath:myData/my_lfads_input.h5}',
        'batch_keys': ['behavior'],
        'attr_keys': ['psth', 'train_cond_idx', 'valid_cond_idx', 'train_decode_mask', 'valid_decode_mask'],
        'batch_size': 256

}

os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
with open(output_yaml, 'w') as f:
    yaml.dump(datamodule_config, f)

print(f"Auto-generated datamodule config at {output_yaml}")
