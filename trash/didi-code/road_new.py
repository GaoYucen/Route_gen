#%%
from tqdm import tqdm
import pandas as pd

#%% 读取link信息
def read_link_from_files(file_names):
    link_df = []
    for file_name in file_names:
        with open('data/link_feature/'+file_name, 'r') as file:
            for line in file:
                # 去除行末的换行符并按逗号分割
                columns = line.strip().split(',')
                link_df.append([int(columns[0]),  int(columns[9]), int(columns[10]), float(columns[1])/float(columns[2])])
    return link_df

# 示例调用，这里假设文件名为 file1.txt, file2.txt 等，你可以根据实际情况修改
file_names = ['part-00000', 'part-00001', 'part-00002', 'part-00003', 'part-00004']
link_df = read_link_from_files(file_names)

#%% 读取link_map信息
def read_link_map_from_files(file_names):
    link_map_df = {}
    for file_name in file_names:
        with open('data/link_map/'+file_name, 'r') as file:
            for line in file:
                columns = line.strip().split(':')
                link_map_df[int(columns[1])] = int(columns[0])
    return link_map_df

# 示例调用，这里假设文件名为 file1.txt, file2.txt 等，你可以根据实际情况修改
file_names = ['part-00000', 'part-00001', 'part-00002', 'part-00003', 'part-00004']
link_map_df = read_link_map_from_files(file_names)

#%% 将link_df中的第0列替换为link_map_df中对应的值
def replace_link_id(link_df, link_map_df):
    for i in range(len(link_df)):
        link_df[i][0] = link_map_df[link_df[i][0]]
    return link_df

link_df = replace_link_id(link_df, link_map_df)

#%% 存储link_df到link_new.npy
link_df = pd.DataFrame(link_df)
link_df.to_csv('data/link_new.csv', index=False, header=False)