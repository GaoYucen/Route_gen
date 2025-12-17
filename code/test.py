#%% 查看data/chengdu_data/[train, valid, test]_data_small_sc.pkl的长度

import pickle

with open('data/chengdu_data/train_data_small_sc.pkl', 'rb') as f:
    train_data = pickle.load(f)
    print(len(train_data))

with open('data/chengdu_data/valid_data_small_sc.pkl', 'rb') as f:
    valid_data = pickle.load(f)
    print(len(valid_data))

with open('data/chengdu_data/test_data_small_sc.pkl', 'rb') as f:
    test_data = pickle.load(f)
    print(len(test_data))


import pickle
import os

city_name = "chengdu_data"  # 请根据需要修改城市名称

def get_pkl_data_length(pkl_file_path):
    """
    读取pickle文件，返回文件中存储对象的元素数量（假设存储的是列表）
    :param pkl_file_path: pickle文件的完整路径
    :return: 元素数量（若文件不存在或格式错误，返回-1）
    """
    # 检查文件是否存在
    if not os.path.exists(pkl_file_path):
        print(f"错误：文件 {pkl_file_path} 不存在！")
        return -1
    
    try:
        # 读取pickle文件
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 检查数据类型（若不是列表，需根据实际类型调整统计方式）
        if isinstance(data, list):
            length = len(data)
            print(f"文件 {os.path.basename(pkl_file_path)} 的数据量：{length} 条")
            return length
        else:
            print(f"警告：文件 {os.path.basename(pkl_file_path)} 存储的不是列表，而是 {type(data)} 类型")
            # 若为字典，可统计键的数量（根据实际需求调整）
            if isinstance(data, dict):
                print(f"字典的键数量：{len(data.keys())}")
                return len(data.keys())
            return -1
    
    except Exception as e:
        print(f"读取文件 {os.path.basename(pkl_file_path)} 时出错：{str(e)}")
        return -1

# -------------------------- 请根据你的实际路径修改以下两行 --------------------------
test_data_path = 'preprocessed/'+city_name+'/test_data_samples.pkl'  # 替换为test_data_small_sc.pkl的路径
selected_points_path = 'preprocessed/'+city_name+'/test_selected_points.pkl'  # 替换为test_selected_points.pkl的路径
# ----------------------------------------------------------------------------------

# 统计两个文件的数据量
print("=" * 50)
get_pkl_data_length(test_data_path)
print("=" * 50)
get_pkl_data_length(selected_points_path)
