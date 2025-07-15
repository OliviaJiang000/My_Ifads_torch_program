import numpy as np
import h5py
import pynwb
import yaml
import os
from pynwb import NWBHDF5IO
from tqdm import tqdm
import lindi

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
    cond_fields = ["trial_type", "trial_version"],  # psth_params
    behavior_field="hand_vel",
    align_field="move_onset_time",
    align_range=(-0.25, 0.45)
):
    fixed_num_bins = int(np.floor((align_range[1]-align_range[0])/bin_size))

    # # 加载远程 nwb 文件
    # f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
    # nwbfile = NWBHDF5IO(file=f, mode='r').read()

    # 打开 NWB 文件
    io = NWBHDF5IO(nwb_path, 'r')
    nwbfile = io.read()

    # Trial 信息
    trials = nwbfile.trials.to_dataframe().reset_index()
    align_times = trials[align_field].values
    start_times = align_times + align_range[0]
    stop_times = align_times + align_range[1]
    num_trials = len(start_times)

    # ======== 获取 trials 表并筛选 ========
    train_trials_df = trials[trials["split"] == "train"].reset_index(drop=True)
    valid_trials_df = trials[trials["split"] == "val"].reset_index(drop=True)


    # 神经元单位（单位数量）
    num_units = len(nwbfile.units)
    spike_tensor_full = np.zeros((num_trials, fixed_num_bins, num_units), dtype=np.float32)
    spike_tensor_train= np.zeros((len(train_trials_df), fixed_num_bins, num_units), dtype=np.float32)
    spike_tensor_valid = np.zeros((len(valid_trials_df), fixed_num_bins, num_units), dtype=np.float32)
    # spike_times 是稀疏结构
    for unit in tqdm(range(num_units), desc="Binning spikes"):
        spike_times_unit = np.array(nwbfile.units["spike_times"][unit])
        for i in range(num_trials):
            t0 = start_times[i]
            t1 = stop_times[i]
            bin_edges = np.linspace(t0, t1, fixed_num_bins + 1)
            mask = (spike_times_unit >= t0) & (spike_times_unit < t1)
            trial_spike_times = spike_times_unit[mask] - t0
            counts, _ = np.histogram(trial_spike_times, bins=bin_edges)
            spike_tensor_full[i, :fixed_num_bins, unit] = counts

    # held-in 神经元索引
    heldout_mask = np.array(nwbfile.units["heldout"].data[:])
    heldin_indices = np.where(heldout_mask == False)[0]
    np.save(h5_output_path.replace(".h5", "_heldin_indices.npy"), heldin_indices)

    # 行为数据
    behav_ts = nwbfile.processing["behavior"][behavior_field]
    behav_data = behav_ts.data[:]
    behav_times = behav_ts.timestamps[:]
    dim = behav_data.shape[1]
    behavior_tensor = np.zeros((num_trials, fixed_num_bins, dim), dtype=np.float32)

    for i in tqdm(range(num_trials), desc="Binning behavior"):
        t0 = start_times[i]
        t1 = stop_times[i]
        bin_edges = np.arange(t0, t1 + bin_size, bin_size)
        mask = (behav_times >= t0) & (behav_times < t1)
        trial_data = behav_data[mask]
        trial_times = behav_times[mask] - t0
        for d in range(dim):
            digitized = np.digitize(trial_times, bin_edges) - 1
            for t_idx in range(len(trial_times)):
                if 0 <= digitized[t_idx] < fixed_num_bins:
                    behavior_tensor[i, digitized[t_idx], d] = trial_data[t_idx, d]
    behavior_tensor_train= behavior_tensor[train_trials_df.index]
    behavior_tensor_valid= behavior_tensor[valid_trials_df.index]
    # 构造 train/valid 数据
    spike_tensor_enc_train = spike_tensor_train[:, :, heldin_indices]
    spike_tensor_enc_valid = spike_tensor_valid[:, :, heldin_indices]
    spike_tensor_recon_train = spike_tensor_train
    spike_tensor_recon_valid = spike_tensor_valid

    recon_mask = np.zeros(num_units, dtype=bool)
    recon_mask[heldin_indices] = True
    train_recon_mask = np.tile(recon_mask, (spike_tensor_train.shape[0], 1))
    valid_recon_mask = np.tile(recon_mask, (spike_tensor_valid.shape[0], 1))


    # 获取train_cond_idx, val_cond_idx
    # ======== 生成 condition 编号（train）========
    train_cond_keys = list(zip(*[train_trials_df[f] for f in cond_fields]))
    train_unique_keys = sorted(set(train_cond_keys))
    train_key_to_idx = {k: i for i, k in enumerate(train_unique_keys)}
    train_cond_idx = np.array([train_key_to_idx[k] for k in train_cond_keys], dtype=np.int32)

    # ======== 生成 condition 编号（valid）========
    valid_cond_keys = list(zip(*[valid_trials_df[f] for f in cond_fields]))
    valid_unique_keys = sorted(set(valid_cond_keys))
    valid_key_to_idx = {k: i for i, k in enumerate(valid_unique_keys)}
    valid_cond_idx = np.array([valid_key_to_idx[k] for k in valid_cond_keys], dtype=np.int32)


    # 保存数据
    with h5py.File(h5_output_path, "w") as f:
        f.create_dataset("train_encod_data", data=spike_tensor_enc_train.astype(np.float16))
        f.create_dataset("valid_encod_data", data=spike_tensor_enc_valid.astype(np.float16))
        f.create_dataset("train_recon_data", data=spike_tensor_recon_train.astype(np.float16))
        f.create_dataset("valid_recon_data", data=spike_tensor_recon_valid.astype(np.float16))
        f.create_dataset("train_behavior", data=behavior_tensor_train.astype(np.float32))
        f.create_dataset("valid_behavior", data=behavior_tensor_valid.astype(np.float32))
        f.create_dataset("train_decode_mask", data=np.ones((num_trials, 1), dtype=bool))
        f.create_dataset("valid_decode_mask", data=np.ones((num_trials, 1), dtype=bool))
        f.create_dataset("train_recon_mask", data=train_recon_mask)
        f.create_dataset("valid_recon_mask", data=valid_recon_mask)
        f.create_dataset("train_cond_idx", data=train_cond_idx)
        f.create_dataset("valid_cond_idx", data=valid_cond_idx)
        f.create_dataset("psth", data=np.mean(spike_tensor_full, axis=0).astype(np.float32))

    print(f"✅ H5 文件保存至 {h5_output_path}")


# 示例运行
if __name__ == "__main__":
    '''
    该版本掩码和idx和psth还没有准备好,recon的time是45还不知道为啥
    '''
    # 参数配置
    nwb_path = '/Users/jojo/Documents/PythonProject/My_IFads_torch_program/nlb_tools/examples/tutorials/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb'  # mc_maze
    output_h5 = 'myData/my_000128_test03.h5'
    output_yaml = 'configs/datamodule/my_datamodule03_000128.yaml'
    bin_size_ms = 20  # 20ms bin

    convert_nwb_to_lfads_h5(
        nwb_path=nwb_path,
        h5_output_path=output_h5,
        bin_size=bin_size_ms / 1000.0,
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