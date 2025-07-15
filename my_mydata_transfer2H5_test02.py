import numpy as np
import h5py
import pynwb
import yaml
import os
from pynwb import NWBHDF5IO
from tqdm import tqdm

PARAMS = {
    'mc_maze': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 100,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 70,
        },
    }}
def convert_nwb_to_lfads_h5(
    nwb_path,
    h5_output_path,
    bin_size=0.02,
    valid_ratio=0.2,
    behavior_field="hand_pos"
):
    # 打开 NWB 文件
    io = NWBHDF5IO(nwb_path, 'r')
    nwbfile = io.read()

    spike_times_table = nwbfile.units["spike_times"].data[:]
    num_units = len(spike_times_table)

    trials = nwbfile.trials.to_dataframe().reset_index()
    start_times = trials["start_time"].values
    stop_times = trials["stop_time"].values
    num_trials = len(start_times)

    max_duration = np.max(stop_times - start_times)
    num_bins = int(np.ceil(max_duration / bin_size))

    # spike binned
    spike_tensor = np.zeros((num_trials, num_bins, num_units), dtype=np.float32)

    for i in tqdm(range(num_trials), desc="Binning spikes"):
        t_start = start_times[i]
        t_stop = stop_times[i]
        bin_edges = np.arange(t_start, t_stop + bin_size, bin_size)
        for unit_idx, unit_spike_train in enumerate(spike_times_table):
            spikes = unit_spike_train[(unit_spike_train >= t_start) & (unit_spike_train < t_stop)] - t_start
            counts, _ = np.histogram(spikes, bins=bin_edges)
            spike_tensor[i, :len(counts), unit_idx] = counts

    # 行为数据
    behav = nwbfile.processing["behavior"][behavior_field]
    behav_data = behav.data[:]
    behav_times = behav.timestamps[:]
    behavior_tensor = np.zeros((num_trials, num_bins, behav_data.shape[1]), dtype=np.float32)

    for i in tqdm(range(num_trials), desc="Extracting behavior"):
        t_start = start_times[i]
        t_stop = stop_times[i]
        bin_edges = np.arange(t_start, t_stop + bin_size, bin_size)
        trial_mask = (behav_times >= t_start) & (behav_times < t_stop)
        trial_data = behav_data[trial_mask]
        trial_times = behav_times[trial_mask] - t_start
        for dim in range(behav_data.shape[1]):
            digitized = np.digitize(trial_times, bin_edges) - 1
            for t_idx in range(len(trial_times)):
                if 0 <= digitized[t_idx] < num_bins:
                    behavior_tensor[i, digitized[t_idx], dim] = trial_data[t_idx, dim]

    # 分 train / valid
    split = int(num_trials * (1 - valid_ratio))
    print(split)
    print(spike_tensor.shape)
    print(nwbfile.units["heldout"].data)
    # 获取 heldout mask
    heldout_mask = np.array(nwbfile.units['heldout'].data[:])  # shape: (182,)
    heldin_indices = np.where(heldout_mask == False)[0]  # 找出 held-in 神经元索引

    print(f"✅ Held-in 神经元数量: {len(heldin_indices)}")

    train_enc = spike_tensor[:split]
    valid_enc = spike_tensor[split:]
    train_behav = behavior_tensor[:split]
    valid_behav = behavior_tensor[split:]

    with h5py.File(h5_output_path, "w") as f:
        f.create_dataset("train_encod_data", data=train_enc.astype(np.float16))
        f.create_dataset("train_recon_data", data=train_enc.astype(np.float16))
        f.create_dataset("valid_encod_data", data=valid_enc.astype(np.float16))
        f.create_dataset("valid_recon_data", data=valid_enc.astype(np.float16))
        f.create_dataset("train_behavior", data=train_behav.astype(np.float32))
        f.create_dataset("valid_behavior", data=valid_behav.astype(np.float32))
        f.create_dataset("train_decode_mask", data=np.ones((train_enc.shape[0], 1), dtype=bool))
        f.create_dataset("valid_decode_mask", data=np.ones((valid_enc.shape[0], 1), dtype=bool))
        f.create_dataset("train_cond_idx", data=np.arange(train_enc.shape[0]))
        f.create_dataset("valid_cond_idx", data=np.arange(valid_enc.shape[0]))
        f.create_dataset("psth", data=np.mean(valid_enc, axis=0))  # 简化平均响应作为 psth

    print(f"✅ 成功写入 LFADS-Torch .h5 文件：{h5_output_path}")
    io.close()


# 示例运行
if __name__ == "__main__":
    # 参数配置
    nwb_path = '/Users/jojo/Documents/PythonProject/My_IFads_torch_program/nlb_tools/examples/tutorials/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb'  # mc_maze
    output_h5 = 'myData/my_000128_test02.h5'
    output_yaml = 'configs/datamodule/my_datamodule02_000128.yaml'
    bin_size_ms = 20  # 20ms bin

    convert_nwb_to_lfads_h5(
        nwb_path=nwb_path,
        h5_output_path=output_h5,
        bin_size=bin_size_ms / 1000.0,
        valid_ratio=0.2,
        behavior_field="hand_pos"
    )


# # 保存 h5
# with h5py.File(output_h5, 'w') as f:
#     f.create_dataset('train_encod_data', data=train_data.astype(np.float32))
#     f.create_dataset('train_recon_data', data=train_data.astype(np.float32))
#     f.create_dataset('valid_encod_data', data=valid_data.astype(np.float32))
#     f.create_dataset('valid_recon_data', data=valid_data.astype(np.float32))
#
# io.close()
# print(f"Saved to {output_h5}")

# # 自动生成 datamodule.yaml
# seq_len = train_data.shape[1]
# datamodule_config = {
#     'datamodule': {
#         '_target_': 'lfads_torch.datamodules.lfads_datamodule.LFADSDataModule',
#         'dataset_path': output_h5,
#         'seq_len': seq_len,
#         'batch_size': 64,
#         'num_workers': 4,
#         'spike_data_transform': {
#             '_target_': 'lfads_torch.datamodules.transforms.SqrtTransform',
#             'offset': 0.1
#         }
#     }
# }
#
# os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
# with open(output_yaml, 'w') as f:
#     yaml.dump(datamodule_config, f)
#
# print(f"Auto-generated datamodule config at {output_yaml}")
