#%% 读取graph_sc.pkl
import pickle
import networkx as nx

import sys
import os

# 获取 code 文件夹的路径
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 code 文件夹路径添加到 sys.path
sys.path.append(code_dir)

from config import get_config

config, _ = get_config()

# 读取节点和边
city_name = config.city

with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()

#%% 用Node2Vec算法生成节点embedding
from node2vec import Node2Vec
from gensim.models import KeyedVectors

node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=4)

model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 保存模型
model.wv.save_word2vec_format('preprocessed/'+city_name+'/node2vec_sc.emb')

# 加载模型
model = KeyedVectors.load_word2vec_format('preprocessed/'+city_name+'/node2vec_sc.emb')


#%%
# 得到所有点的嵌入
embeddings = {}
for node in G.nodes():
    embeddings[node] = model[str(node)]

# # 将嵌入转换为numpy数组
# embeddings_array = np.array([embeddings[node] for node in G.nodes()])

# # 打印嵌入结果
# print(embeddings_array)

#%% 保存嵌入结果
# np.save('data/'+city_name+'/embeddings.npy', embeddings_array)

# 保存嵌入结果
import pickle

with open('preprocessed/'+city_name+'/node_embedding_sc.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
    f.close()