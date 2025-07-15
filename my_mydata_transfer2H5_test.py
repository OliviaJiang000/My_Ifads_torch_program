import numpy as np
import h5py
import pynwb
import yaml
import os

# 参数配置
nwb_path = 'nlb_tools/examples/tutorials/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb'   # mc_maze
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

# 计算整体 bin 数
session_end_time = end_times.max()
bin_size_sec = bin_size_ms / 1000
n_total_bins = int(np.ceil(session_end_time / bin_size_sec))

# 整体 binning
spike_counts = np.zeros((n_units, n_total_bins), dtype=np.int16)

for i in range(n_units):
    spikes = units['spike_times'][i]  # 使用正确读取方式
    if spikes.size == 0:
        continue
    bin_idx = (spikes / bin_size_sec).astype(int)
    for idx in np.atleast_1d(bin_idx):
        if idx < n_total_bins:
            spike_counts[i, idx] += 1

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

def slice_trials(unit_indices):
    data = []
    for start_bin, end_bin in trial_slices:
        trial_bins = spike_counts[unit_indices, start_bin:end_bin]
        pad_width = max_trial_bins - trial_bins.shape[1]
        if pad_width > 0:
            trial_bins = np.pad(trial_bins, ((0,0), (0,pad_width)))
        data.append(trial_bins.T)  # 转置成 [time, unit]
    return np.array(data)

train_data = slice_trials(heldin_units)
valid_data = slice_trials(heldout_units)

print("Train data shape:", train_data.shape)
print("Valid data shape:", valid_data.shape)

# 保存 h5
with h5py.File(output_h5, 'w') as f:
    f.create_dataset('train_encod_data', data=train_data.astype(np.float32))
    f.create_dataset('train_recon_data', data=train_data.astype(np.float32))
    f.create_dataset('valid_encod_data', data=valid_data.astype(np.float32))
    f.create_dataset('valid_recon_data', data=valid_data.astype(np.float32))

io.close()
print(f"Saved to {output_h5}")

# 自动生成 datamodule.yaml
seq_len = train_data.shape[1]
datamodule_config = {
    'datamodule': {
        '_target_': 'lfads_torch.datamodules.lfads_datamodule.LFADSDataModule',
        'dataset_path': output_h5,
        'seq_len': seq_len,
        'batch_size': 64,
        'num_workers': 4,
        'spike_data_transform': {
            '_target_': 'lfads_torch.datamodules.transforms.SqrtTransform',
            'offset': 0.1
        }
    }
}

os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
with open(output_yaml, 'w') as f:
    yaml.dump(datamodule_config, f)

print(f"Auto-generated datamodule config at {output_yaml}")
