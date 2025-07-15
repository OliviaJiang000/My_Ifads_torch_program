import h5py
import numpy as np

# 修改为你自己的路径
# input data
file_path='datasets/mc_maze_medium-05ms-val.h5'
file_path='datasets/dmfc_rsg-05ms-val.h5'
# file_path='datasets/area2_bump-05ms-val.h5'
# file_path='datasets/mc_rtt-05ms-val.h5'
# file_path='datasets/mc_maze_large-05ms-val.h5'
# file_path='datasets/mc_maze-20ms-val.h5'

# output data
# file_path='scripts/runs/lfads-torch-example/nlb_mc_maze/250704_exampleSingle/lfads_output_mc_maze-20ms-val.h5'
file_path='myData/my_000128_test04.h5'
file_path='myData/myTry_000128_5.h5'
# file_path='/Users/jojo/Documents/PythonProject/My_IFads_torch_program/lfads-torch/scripts/runs/my-try-000128/my_datamodule01/250706_exampleMine/lfads_output_myTry_000128_5.h5'
def print_h5_structure(filepath):
    import h5py

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

    with h5py.File(filepath, 'r') as f:
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

print_h5_structure(file_path)
