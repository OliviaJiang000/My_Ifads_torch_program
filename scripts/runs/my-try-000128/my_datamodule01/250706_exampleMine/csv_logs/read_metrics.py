import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
csv_file_path = 'metrics.csv'
df = pd.read_csv(csv_file_path)

# 查看数据的基本信息
print("数据形状:", df.shape)
print("\n列名:")
print(df.columns.tolist())

# 查看前几行数据
print("\n前5行数据:")
print(df.head())

# 查看数据类型和缺失值
print("\n数据信息:")
print(df.info())

# 查看数值型列的统计信息
print("\n数值型列统计:")
print(df.describe())

# ['lr-AdamW', 'step', 'valid/recon/sess0', 'valid/loss', 'valid/recon', 'valid/bps', 'valid/co_bps', 'valid/fp_bps', 'valid/r2', 'valid/wt_l2', 'valid/wt_l2/ramp', 'valid/wt_kl', 'valid/wt_kl/ic', 'valid/wt_kl/co', 'valid/wt_kl/ramp', 'valid/recon_smth', 'hp_metric', 'cur_epoch', 'hp/lr_init', 'hp/dropout_rate', 'hp/l2_ic_enc_scale', 'hp/l2_ci_enc_scale', 'hp/l2_gen_scale', 'hp/l2_con_scale', 'hp/kl_co_scale', 'hp/kl_ic_scale', 'hp/weight_decay', 'hp/cd_rate', 'epoch', 'train/recon/sess0', 'train/loss', 'train/recon', 'train/bps', 'train/co_bps', 'train/fp_bps', 'train/r2', 'train/wt_l2', 'train/wt_l2/ramp', 'train/wt_kl', 'train/wt_kl/ic', 'train/wt_kl/co', 'train/wt_kl/ramp']
print(df[['valid/recon/sess0','valid/co_bps','train/fp_bps',]])