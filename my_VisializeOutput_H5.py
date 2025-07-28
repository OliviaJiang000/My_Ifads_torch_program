import h5py
import numpy as np
import matplotlib.pyplot as plt

# 修改为你自己的路径
# input data
file_path='datasets/mc_maze_medium-05ms-val.h5'
file_path='datasets/dmfc_rsg-05ms-val.h5'
# file_path='datasets/area2_bump-05ms-val.h5'
# file_path='datasets/mc_rtt-05ms-val.h5'
# file_path='datasets/mc_maze_large-05ms-val.h5'
file_path='datasets/mc_maze-20ms-val.h5'

# output data
# file_path='scripts/runs/lfads-torch-example/nlb_mc_maze/250704_exampleSingle/lfads_output_mc_maze-20ms-val.h5'
# file_path='myData/my_000128_test04.h5'
# file_path='myData/myTry_000128_5.h5'
# file_path='myData/myTry_000128_5_addMask.h5'
# file_path='scripts/runs/lfads-torch-example/nlb_mc_maze/250704_exampleSingle/lfads_output_mc_maze-20ms-val.h5'
# file_path='datasets/mc_maze-20ms-val.h5'
file_path='scripts/runs/my-try-000128/my_datamodule01/250723_exampleMine/lfads_output_myTry_000128_5_addMask.h5'
def print_h5_structure(f):
    '''
    获得h5文件的数据结构
    :param filepath:
    :return:
    '''


    description_map = {
        "train_encod_data": "训练输入（编码器输入）",
        "train_recon_data": "训练重建目标",
        "valid_encod_data": "验证输入",
        "valid_recon_data": "验证目标",
        "train_behavior": "行为变量（例如手的位置）",
        "valid_behavior": "验证行为",
        "train_decode_mask": "解码时使用的掩码",
        "valid_decode_mask": "解码时使用的掩码",
        "train_cond_idx": "条件标签",
        "valid_cond_idx": "条件标签",
        "psth": "平均神经反应（条件平均）",
    }

    print("=" * 90)
    print(f"{'字段名':<25}{'形状':<20}{'数据类型':<10}{'说明':<30}")
    print("=" * 90)
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            shape = str(obj.shape)
            dtype = str(obj.dtype)
            desc = description_map.get(key, "")
            print(f"{key:<25} {shape:<20} {dtype:<10} {desc:<30} ")



def plot_raster_h5(f, dataset_name='train_encod_data',neuro_idx=1,t_before=0, t_after=None, bin_size=20):
    """
    Reads spike-count data from HDF5 and draws a raster plot for one neuro.

    Parameters
    ----------
    file_path : str
        Path to the .h5 file.
    dataset_name : str
        Name of the dataset (e.g., 'train_encod_data').
    neuro : int
        neuro index to plot (0-based).
    t_before : float
        Time (in same units as bins) before bin 0 to include (default 0).
    t_after : float or None
        Time after the final bin to include; if None, covers full length.
    bin_size : float
        (/ms) Duration of each time bin (in seconds or arbitrary units).
    """
    # Open HDF5 and load dataset

    data = f[dataset_name][:]  # shape: (n_trials, n_timebins, n_neuros)
    # print(data)
    # print(data.shape)
    # print(data[0].shape)
    n_trials, n_timebins, n_neuros = data.shape
    if neuro_idx < 0 or neuro_idx >= n_neuros:
        raise ValueError(f"neuro must be in [0, {n_neuros-1}]")

    # Determine time axis
    if t_after is None:
        t_after = n_timebins * bin_size
    time_bins = np.arange(-t_before, t_after, bin_size)

    # Prepare plot
    plt.figure(figsize=(8, 6))

    # Loop over each trial

    for i in range(n_trials):
        # Extract counts for this trial and neuro
        counts = data[i, :, neuro_idx]
        # Find bins where there was at least one spike
        spk_bins = np.where(counts > 0)[0]
        # Compute spike times as bin centers
        spike_times = -t_before + (spk_bins + 0.5) * bin_size
        # Draw a tick for each spike （默认不管spike数值，只显示是否spike了）
        plt.vlines(spike_times, i, i+1, linewidth=20)

    # Formatting
    # plt.grid('minor')
    plt.xlabel('Time relative to trial start (units)')
    plt.ylabel('Trial')
    plt.title(f'Raster plot: neuro {neuro_idx}')
    plt.ylim(0.5, n_neuros + 0.5)
    plt.xlim(-t_before, t_after)
    plt.tight_layout()
    plt.show()



def plot_psth_h5(
    f,
    dataset_name='train_encod_data',
    trial_range=None,
    bin_size=20,
    neuro_idx=0,
    t_start=0,
    t_stop=None
):
    """
    计算并绘制 PSTH：
    - 读取多 trial、单通道或多通道的 spike-count 数据
    - 调用 Elephant 计算合并后的 PSTH
    """
    data = f[dataset_name][:]  # shape: (n_trials, n_timebins, n_neurons)
    n_trials, n_timebins, n_neurons = data.shape

    # Trial 范围
    if trial_range is None:
        trials = np.arange(n_trials)
    else:
        trials = np.array(trial_range)

    # 提取指定神经元所有 trial 的数据
    all_trials = data[trials, :, neuro_idx]  # shape: (n_trials, n_timebins)

    # 判断是否为 count 类型（整数为主）
    is_count_data = np.allclose(all_trials, np.round(all_trials))
    # print(f"Is count data: {is_count_data}")

    if is_count_data:
        all_rates = all_trials / (bin_size * 1e-3)  # Convert to Hz
    else:
        all_rates = all_trials  # Already in Hz

    # 求每个时间点的平均
    mean_rate = np.nanmean(all_rates, axis=0)

    # 时间坐标
    t_stop = t_stop or n_timebins * bin_size
    time_axis = np.arange(0, n_timebins * bin_size, bin_size)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, mean_rate, label=f"Neuron {neuro_idx}")
    plt.xlim(t_start, t_stop)
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing Rate (Hz)")
    plt.title(f"PSTH | {dataset_name} | Neuron {neuro_idx}")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_psth_h5_singleTrial(
    f,
    dataset_name='train_encod_data',
    trial_idx=0,
    neuro_idx=0,
    bin_size=20,
    t_before=0,
    t_after=None
):
    """
    绘制单个 trial、单个神经元的 PSTH。

    参数
    ------
    f : h5py.File
        已打开的 HDF5 文件对象
    dataset_name : str
        数据集名称，例如 'train_encod_data'
    trial_idx : int
        要绘制的 trial 下标（从 0 开始）
    neuro_idx : int
        要绘制的神经元下标（从 0 开始）
    bin_size : float
        每个时间 bin 的宽度（单位为 ms）
    t_before : float
        在起始处预留的时间（ms）
    t_after : float or None
        在末尾预留的时间；若为 None，则使用数据长度决定
    """
    # 读取数据
    data = f[dataset_name][:]  # shape: (n_trials, n_timebins, n_neurons)
    n_trials, n_timebins, n_neurons = data.shape

    # 检查合法性
    if not (0 <= trial_idx < n_trials):
        raise ValueError(f"trial_idx must be in [0, {n_trials - 1}]")
    if not (0 <= neuro_idx < n_neurons):
        raise ValueError(f"neuro_idx must be in [0, {n_neurons - 1}]")

    if t_after is None:
        t_after = n_timebins * bin_size

    # 构造时间轴
    bin_edges = np.arange(-t_before, t_after + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2

    # 提取该 trial、该 neuron 的值
    values = data[trial_idx, :, neuro_idx]

    # 自动判断是 count 还是 rate
    is_count_data = np.allclose(values, np.round(values))
    print(f"[Info] Detected {'count' if is_count_data else 'rate'} data.")

    if is_count_data:
        rates = values / (bin_size * 1e-3)  # Convert count → Hz
    else:
        rates = values  # Already in Hz

    # 绘图
    plt.figure(figsize=(8, 4))
    plt.bar(bin_centers[:len(rates)], rates, width=bin_size,
            align='center', color='C0', edgecolor='k')
    plt.xlabel('Time relative to trial start (ms)')
    plt.ylabel('Firing rate (Hz)')
    plt.title(f'Single-Trial PSTH | {dataset_name} | Trial {trial_idx}, Neuron {neuro_idx}')
    plt.grid(axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# def plot_psth_h5_singleTrial(
#         f,
#         dataset_name='train_encod_data',
#         trial_idx=0,
#         neuro_idx=0,
#         bin_size=20,
#         t_before=0,
#         t_after=None
# ):
#     """
#     绘制单个 trial、单个通道的 PSTH。
#
#     参数
#     ------
#     file_path : str
#         HDF5 文件路径
#     dataset_name : str
#         数据集名称，例如 'train_encod_data'
#     trial_idx : int
#         要绘制的 trial 下标（0-based）
#     neuro_idx : int
#         要绘制的通道/neuronal 单元下标（0-based）
#     bin_size : float
#         每个时间 bin 的长度（同数据单位，例如 ms）
#     t_before : float
#         在第一个 bin 之前要包含的时间
#     t_after : float or None
#         在最后一个 bin 之后要包含的时间；若为 None，则使用完整长度
#     """
#     # 读取数据
#     data = f[dataset_name][:]  # (n_trials, n_timebins, n_neurons)
#
#     n_trials, n_timebins, n_neurons = data.shape
#     if not (0 <= trial_idx < n_trials):
#         raise ValueError(f"trial_idx must be in [0, {n_trials - 1}]")
#     if not (0 <= neuro_idx < n_neurons):
#         raise ValueError(f"neuro_idx must be in [0, {n_neurons - 1}]")
#
#     if t_after is None:
#         t_after = n_timebins * bin_size
#     bin_edges = np.arange(-t_before, t_after + bin_size, bin_size)
#     bin_centers = bin_edges[:-1] + bin_size / 2
#
#     # 提取 counts
#     counts= data[trial_idx, :, neuro_idx]
#     # print(f"Trial {counts}, counts sum: {counts.sum()}")
#     # 计算 firing rate: counts per bin / bin duration -> spikes per ms, *1000 for Hz if ms unit
#     # 假设 bin_size 单位为 ms，此处转换为 seconds
#     # rates = counts / (bin_size * 1e-3)
#     # 自动判断是否是 count 或 rate
#     is_count_data = np.allclose(counts, np.round(counts))
#
#     if is_count_data:
#         rates = counts / (bin_size * 1e-3)  # Hz
#     else:
#         rates = counts  # 已是 Hz
#     # 绘图
#     plt.figure(figsize=(8, 4))
#     plt.bar(bin_centers, rates, width=bin_size, align='center', color='C0', edgecolor='k')
#     plt.xlabel('Time relative to trial start (ms)')
#     plt.ylabel(f'{dataset_name} Firing rate (Hz)')
#     plt.title(f'Single-Trial PSTH (Trial {trial_idx}, Neuro {neuro_idx})')
#     plt.grid(axis='x', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.show()
#
#     # data = f[dataset_name][:]
#     # n_trials, n_bins, n_neurons = data.shape
#     #
#     # # 数据合法性检查
#     # if not (0 <= trial_idx < n_trials):
#     #     raise ValueError(f"trial_idx 应在 [0, {n_trials - 1}] 范围内")
#     # if not (0 <= neuro_idx < n_neurons):
#     #     raise ValueError(f"neuro_idx 应在 [0, {n_neurons - 1}] 范围内")
#     #
#     #
#     # # 横轴时间坐标（单位 ms）
#     # time_axis = np.arange(0, n_bins * bin_size, bin_size)
#     #
#     # # 获取指定 trial 的指定神经元数据
#     # rates = data[trial_idx, :, neuro_idx]
#     #
#     # # 可视化
#     # plt.figure(figsize=(8, 4))
#     # plt.step(time_axis, rates, where='mid', color='blue')
#     # plt.xlabel("Time (ms)")
#     # plt.ylabel("Firing Rate (Hz)")
#     # plt.title(f"Firing Rate | Trial {trial_idx}, Neuron {neuro_idx}")
#     # plt.grid(True, linestyle='--', linewidth=0.5)
#     #
#     # plt.tight_layout()
#     # plt.show()

if __name__ == '__main__':

    with h5py.File(file_path, 'r') as f:

        neuroNo=102
        # 获得h5文件的数据结构
        print_h5_structure(f)
        # raster plot
        plot_raster_h5(f,bin_size=20,neuro_idx=neuroNo,dataset_name='train_encod_data')



        # psth plot
        plot_psth_h5(f,bin_size=20,dataset_name='train_output_params',neuro_idx=neuroNo)
        plot_psth_h5(f, bin_size=20, dataset_name='train_recon_data', neuro_idx=neuroNo)
        plot_psth_h5(f, bin_size=20, dataset_name='valid_output_params', neuro_idx=neuroNo)
        plot_psth_h5(f, bin_size=20, dataset_name='valid_recon_data', neuro_idx=neuroNo)

        # psth plot (single trial)
        plot_psth_h5_singleTrial(f,bin_size=20,dataset_name='train_output_params',trial_idx=1,neuro_idx=neuroNo)
        plot_psth_h5_singleTrial(f, bin_size=20, dataset_name='train_recon_data', trial_idx=1, neuro_idx=neuroNo)
        plot_psth_h5_singleTrial(f, bin_size=20, dataset_name='valid_output_params', trial_idx=1, neuro_idx=neuroNo)
        plot_psth_h5_singleTrial(f, bin_size=20, dataset_name='valid_recon_data', trial_idx=1, neuro_idx=neuroNo)
        # if dataType=='outPut':


