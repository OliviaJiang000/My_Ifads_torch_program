import os
import pickle
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

import numpy as np
import pandas as pd
import h5py

import logging

align = 'holdstart'
length = 0.0
# 20ms一个窗口
bins = [-0.5, 1.5]
logging.basicConfig(level=logging.INFO)


dataset_name = 'mc_maze' # MC_Maze
datapath = '/Users/jojo/Documents/PythonProject/BCI/datasets/CSC8099_SinapsData/spt'

markerPath='/Users/jojo/Documents/PythonProject/BCI/datasets/CSC8099_SinapsData/assignedmarkers.mat'

# dataset = NWBDataset(datapath,split_heldout=True)

import numpy as np
import matplotlib.pyplot as plt

def show_spikes_singleNeuro(spt, bin_size=5, t_before=0, t_after=None):
    plt.eventplot(spt, orientation='horizontal', colors='black')
    plt.xlabel("Time (s)")
    plt.title("Spike Times Raster Plot")
    plt.show()
def read_spt(filename, to_seconds=True):
    spt_list= {}
    neuro_list=[]
    i=0
    for file in os.listdir(filename):
        neuro_name=file.split('-')[1].split('.')[0]
        i+=1
        # 以 int32 方式读取
        spt = np.fromfile(filename + '/' + file, dtype=np.int32)

        if to_seconds:
            spt = spt / 2e5  # 每个单位是5μs，即1秒 = 200000个单位

        spt_list[i]=spt
        neuro_list.append({'neuro_id': i, 'neuro_name': neuro_name})

    # # 存储神经元id与name的对应关系
    # neuro_list=pd.DataFrame(neuro_list)
    # neuro_list.to_csv('/Users/jojo/Documents/PythonProject/BCI/datasets/CSC8099_SinapsData/jyy/neuro_list.csv')

    # # 所有spt数据存至一个文件中
    # import pickle
    # with open('/Users/jojo/Documents/PythonProject/BCI/datasets/CSC8099_SinapsData/jyy/all_spt_list.pkl', "wb") as f:
    #     pickle.dump(spt_list, f)


    return spt_list

from scipy.io import loadmat

def read_markers(mat_file):
    data = loadmat(mat_file)
    markers = {}
    for key in data:

        if not key.startswith('__') and not key=='ntrials':
            if key in ['targetdir','targetdist']:
                markers[key] = data[key].flatten()
            else:
                markers[key] = data[key].flatten() / 25000  # 25kHz → 秒
    return markers


def compute_spike_counts(spt_list, markers):
    num_neurons = len(spt_list)
    num_trials = len(markers)
    num_bins = int((bins[1] - bins[0])/length) # 取2000ms，20ms一个窗口，一共100个窗口。

    spike_counts = np.zeros(( num_trials, num_bins, num_neurons), dtype=int)
    for neuro_idx in spt_list:
        print(f"Processing neuron {neuro_idx}/{num_neurons}...")
        spt = spt_list[neuro_idx]
        for i_trial, row in markers.iterrows():
            # 构建边界
            bin_left = row[align] + bins[0]
            bin_right = row[align] + bins[1]

            # 统计 histogram（左闭右开）
            counts, _ = np.histogram(spt, bins=np.arange(bin_left,bin_right+0.0001,length))
            spike_counts[i_trial,:, neuro_idx-1] = counts



    print(spike_counts.shape)
    return spike_counts

def get_cond_idx(markers):
    cond_all=np.unique(markers['condition'])
    C=len(cond_all)
    cond_idx=np.empty((C,), dtype=object)
    for i,c in enumerate(cond_all):
        cond_idx[i]=np.where(markers['condition']==c)[0]
    return cond_idx

def get_psth(spike_counts,cond_idx):
    C=len(cond_idx)
    psth = np.zeros((C, spike_counts.shape[1], spike_counts.shape[2]), dtype=np.float32)
    i=1
    for cond in range(C):
        print( cond)
        print(spike_counts[cond_idx[cond]].shape)
        psth[cond] = spike_counts[cond_idx[cond]].mean(axis=0)
        # plt.plot(psth[cond][:,1])
        # plt.show()
    return psth

def plot_psth(spike_counts,neuron_idx=0,trials_u=None,cond=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    time_axis = np.arange(0, spike_counts.shape[1] * length, length)

    if trials_u is not None and trials_u.any():
        spike_counts = spike_counts[trials_u,:,neuron_idx]
    else:
        spike_counts = spike_counts[:, :, neuron_idx]


    print(spike_counts.shape)
    mean_rate=np.nanmean(spike_counts,axis=0)/length
    print('mean_rate',mean_rate.shape)
    plt.plot(time_axis, mean_rate,label=f'Average Cond {cond}')
    plt.title(f'Condition PSTH - Cond {cond},Neuro {neuron_idx}')

    df=spike_counts[trials_u[0],:]/length
    plt.plot(time_axis,df,label=f'single trial {trials_u[0]}')
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing Rate (Hz)")

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_raster(spike_counts,neuro_idx,trials_u=None,t_before=0, t_after=None):
    if t_after ==None:
        t_after=t_before+bins[1]-bins[0]
    if trials_u is not None and trials_u.any():
        spike_counts = spike_counts[trials_u,:,:]
    n_trials, n_timebins, n_neuros = spike_counts.shape
    print(spike_counts.shape)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_trials):
        counts = spike_counts[i, :, neuro_idx]
        # 找出有 spike 的 bin 索引
        occupied = np.where(counts > 0)[0]
        # 对应的 bin 左边界
        lefts = -t_before + occupied * length
        # 横向宽度都是 length
        widths = np.full_like(lefts, length)
        # y 位置和高度
        ys = np.full_like(lefts, i)
        ax.barh(ys, widths, left=lefts, height=1.0, color='C0', edgecolor=None)


    # Formatting
    # plt.grid('minor')
    ax.set_xlim(t_before,t_after)
    ax.set_ylim(0.5, n_trials + 0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Trial')
    plt.title(f'Raster plot: neuro {neuro_idx}')
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    Neuro_idx=401
    cond=15
    all_spt_list=read_spt(datapath,True)
    # print(all_spt_list)
    # show_spikes_singleNeuro(all_spt_list[Neuro_idx])

    markers=read_markers(markerPath) # ['holdend', 'holdstart', 'targetdir', 'targetdist', 'trialstart']

    # for col in ['trialstart', 'holdstart', 'holdend']:
    #     plt.plot(markers[col],label= col)
    #     plt.legend()

    markers=pd.DataFrame(markers)
    print(markers)


    markers['condition']=markers.apply(lambda x: f"{int(x['targetdir'])}-{int(x['targetdist'])}",axis=1)
    markers['trial']=markers.index
    markers['timestamps']=markers.apply(lambda x: np.arange(x['holdstart']-0.5,x['holdstart']+1.5,0.02),axis=1)  # 取2000ms，20ms一个窗口，一共100个时间点

    # print(markers)
    # 计算spike_counts
    spike_counts=compute_spike_counts(all_spt_list, markers)

    with open('/Users/jojo/Documents/PythonProject/BCI/datasets/CSC8099_SinapsData/jyy/spike_counts.pkl','wb') as f:
        pickle.dump(spike_counts, f)

    # with open('/Users/jojo/Documents/PythonProject/BCI/datasets/CSC8099_SinapsData/jyy/spike_counts.pkl', 'rb') as f:
    #     spike_counts=pickle.load(f)


    cond_idx=get_cond_idx( markers)
    psth=get_psth(spike_counts, cond_idx)
    # print(psth.shape)

    plot_psth(spike_counts,Neuro_idx-1,trials_u=cond_idx[cond],cond=cond)

    plot_raster(spike_counts,Neuro_idx-1,cond_idx[cond])  # ,cond_idx[cond]

    plot_raster(spike_counts,Neuro_idx-1)



