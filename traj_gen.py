#%%
# 导入一些必要的库
import networkx as nx
import geopandas as gpd
import pickle
import matplotlib.pyplot as plt
import numpy as np

def similarity(traj1, traj2):
    return len(set(traj1).intersection(set(traj2)))

# precision, recall, F1-Score, EDT的definition

city_list = ['beijing', 'chengdu', 'cityindia', 'harbin', 'porto']

for city in city_list:

    print('City:', city)

    node_df = gpd.read_file('preprocessed_data/'+city+'_data/map/nodes.shp')
    edge_df = gpd.read_file('preprocessed_data/'+city+'_data/map/edges.shp')

    #%% 使用edge_df构造networkx的DiGraph
    G = nx.DiGraph()
    for i in range(len(edge_df)):
        G.add_edge(edge_df.iloc[i]['u'], edge_df.iloc[i]['v'], weight = 1.2, index=i)

    #%% 轨迹数据
    file_name = 'preprocessed_data/'+city+'_data/preprocessed_train_trips_small.pkl'

    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        f.close()

    # 提取data的全部第一列形成trajs
    trajs = [row[1] for row in data]

    trajs_num = 1000
    trajs = trajs[0:trajs_num]

    #%%
    # 根据历史轨迹计算每条边的权重，表示该边被选择的概率
    # 这里使用了一个简单的方法，即统计每条边出现的次数，然后除以总次数
    # 也可以使用其他的方法，例如基于马尔可夫链或者循环神经网络等
    edge_count = {}
    total_count = 0
    for traj in trajs:
        for edge in traj:
            edge_count[edge] = edge_count.get(edge, 0) + 1
            total_count += 1

    #%%
    for edge_index in edge_count:
        G.edges[edge_df.iloc[edge_index]['u'], edge_df.iloc[edge_index]['v']]['weight'] = 1/edge_count[edge_index]

    #%% 读取mid_point_list
    mid_point_list = []
    with open('param/'+city+'_mid_point_list.txt', 'r') as f:
        for line in f.readlines():
            mid_point_list.append(int(line))
        f.close()

    #%%
    # 根据起点和终点，使用networkx库中的函数，找到一条最短路径，表示推荐的轨迹
    # 这里使用了一个基于权重的方法，即选择权重最大的边作为下一步
    # 也可以使用其他的方法，例如基于随机选择或者贪心算法等
    if_mid_list = []
    similar_list = []
    similar_list_mid = []
    for i in range(0, 1000):
        traj = trajs[i]
        mid_edge = mid_point_list[i]

        start_point = edge_df.iloc[traj[0]]['v']
        end_point = edge_df.iloc[traj[-1]]['u']
        mid_point_1 = edge_df.iloc[mid_edge]['u']
        mid_point_2 = edge_df.iloc[mid_edge]['v']
        try:
            recommended_trajectory = nx.dijkstra_path(G, start_point, end_point, weight='weight')
            recommended_trajectory_1 = nx.dijkstra_path(G, start_point, mid_point_1, weight='weight')
            recommended_trajectory_2 = nx.dijkstra_path(G, mid_point_2, end_point, weight='weight')
            # 判断mid_point是否在traj中
            if mid_edge in traj:
                if_mid_list.append(True)
            else:
                if_mid_list.append(False)
            pass
        except Exception as e:
            print(f"发生异常：{e}")
            continue
        recommended_trajectory_path = [G.edges[recommended_trajectory[i], recommended_trajectory[i+1]]['index'] for i in range(len(recommended_trajectory)-1)]
        # 补充起始边和最后一条边
        recommended_trajectory_path = [traj[0]] + recommended_trajectory_path + [traj[-1]]
        # 计算推荐的轨迹和原始轨迹的相似度
        similar_value = similarity(recommended_trajectory_path, traj)
        similar_list.append(similar_value)
        # print("Similarity:", similar_value)

        # 有中间点的情况
        recommended_trajectory_path_1 = [G.edges[recommended_trajectory_1[i], recommended_trajectory_1[i+1]]['index'] for i in range(len(recommended_trajectory_1)-1)]
        recommended_trajectory_path_2 = [G.edges[recommended_trajectory_2[i], recommended_trajectory_2[i+1]]['index'] for i in range(len(recommended_trajectory_2)-1)]
        # 补充起始边和最后一条边
        recommended_trajectory_path = [traj[0]] + recommended_trajectory_path_1 + [mid_edge] + recommended_trajectory_path_2 + [traj[-1]]
        # 计算推荐的轨迹和原始轨迹的相似度
        similar_mid_value = similarity(recommended_trajectory_path, traj)
        similar_list_mid.append(similar_mid_value)
        # print("Similarity (mid):", similar_mid_value)

    #%% 统计if_mid_list中1的个数
    print('if_mid_list:', sum(if_mid_list))

    print('Similarity mean:', sum(similar_list)/len(similar_list))
    print('Similarity (mid) mean:', sum(similar_list_mid)/len(similar_list_mid))
    print('Similarity gap ratio:', (sum(similar_list_mid)/len(similar_list_mid))/(sum(similar_list)/len(similar_list))-1)

    # #%% 计算轨迹平均长度
    # traj_length_list = []
    # for traj in trajs:
    #     traj_length_list.append(len(traj))
    # print('Trajectory length mean:', sum(traj_length_list)/len(traj_length_list))

    #%% 绘制对比similarity_list和similarity_mid_list的柱状图
    length = 50
    x = np.arange(length)
    width = 0.4
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2 + 1, similar_list[:length], width, label='Similarity')
    rects2 = ax.bar(x + width/2 + 1, similar_list_mid[:length], width, label='Similarity (mid)')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity vs Similarity with mid point')
    ax.legend()
    fig.tight_layout()
    plt.savefig('figure/'+city+'_similarity.pdf')

    #%%
    # 仅对比if_mid_list为1位置的similarity和similarity_mid
    similar_list = np.array(similar_list)
    similar_list_mid = np.array(similar_list_mid)
    similar_list_z = similar_list[if_mid_list]
    similar_list_mid_z = similar_list_mid[if_mid_list]
    print('len of Similarity z mean:', len(similar_list_z))
    print('len of Similarity (mid) z mean:', len(similar_list_mid_z))
    inverse_if_mid_list = [not i for i in if_mid_list]
    similar_list_f = similar_list[inverse_if_mid_list]
    similar_list_mid_f = similar_list_mid[inverse_if_mid_list]
    print('len of Similarity f mean:', len(similar_list_f))
    print('len of Similarity (mid) f mean:', len(similar_list_mid_f))


    print('Similarity z mean:', sum(similar_list_z)/len(similar_list_z))
    print('Similarity (mid) z mean:', sum(similar_list_mid_z)/len(similar_list_mid_z))
    print('Similarity z gap ratio:', (sum(similar_list_mid_z) / len(similar_list_mid_z)) / (sum(similar_list_z) / len(similar_list_z)) - 1)
    print('Similarity f mean:', sum(similar_list_f)/len(similar_list_f))
    print('Similarity (mid) f mean:', sum(similar_list_mid_f)/len(similar_list_mid_f))
    print('Similarity f gap ratio:', (sum(similar_list_mid_f) / len(similar_list_mid_f)) / (sum(similar_list_f) / len(similar_list_f)) - 1)

    #%% 绘制对比similarity_list和similarity_mid_list的柱状图
    length = 50
    x = np.arange(length)
    width = 0.4
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2 + 1, similar_list_z[:length], width, label='Similarity')
    rects2 = ax.bar(x + width/2 + 1, similar_list_mid_z[:length], width, label='Similarity (mid)')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity vs Similarity with mid point')
    ax.legend()
    fig.tight_layout()
    plt.savefig('figure/'+city+'_similarity_z.pdf')

    #%% 绘制对比similarity_list和similarity_mid_list的柱状图
    length = 50
    x = np.arange(length)
    width = 0.4
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2 + 1, similar_list_f[:length], width, label='Similarity')
    rects2 = ax.bar(x + width/2 + 1, similar_list_mid_f[:length], width, label='Similarity (mid)')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity vs Similarity with mid point')
    ax.legend()
    fig.tight_layout()
    plt.savefig('figure/'+city+'_similarity_f.pdf')














# # 可视化地图信息和推荐的轨迹，使用matplotlib库
# # 这里使用了一个简单的方法，即根据点的编号，为每个点分配一个固定的位置
# # 也可以使用其他的方法，例如基于真实的经纬度或者随机的位置等
# pos = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (1.5, 1), 4: (0.5, 2), 5: (2.5, 2)}
# nx.draw_networkx_nodes(graph, pos, node_color='lightblue')
# nx.draw_networkx_edges(graph, pos, edge_color='gray')
# nx.draw_networkx_labels(graph, pos)
# nx.draw_networkx_edges(graph, pos, edgelist=[(recommended_trajectory[i], recommended_trajectory[i+1]) for i in range(len(recommended_trajectory)-1)], edge_color='red', width=3)
# plt.axis('off')
# plt.show()
