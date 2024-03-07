#%%
import numpy as np
import pickle

city_list = ['beijing', 'chengdu', 'cityindia', 'harbin', 'porto']

for city in city_list:

    file_name = 'preprocessed_data/'+city+'_data/preprocessed_train_trips_small.pkl'

    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        f.close()

    #%% 提取data的全部第一列形成trajs
    trajs = [row[1] for row in data]

    #%%
    trajs_num = 1000
    trajs = trajs[0:trajs_num]

    #%% 最长的traj
    print('max traj: ', max([len(i) for i in trajs]))

    #%% 统计平均traj长度
    print('mean traj: ', sum([len(i) for i in trajs])/len(trajs))

    #%% 统计trajs中有多少个不同的数字
    links = set([j for i in trajs for j in i])
    print('No. of edges: ', len(links))

    #%% links中最大的元素
    max_link = max(links) + 1
    print('Max edge index:', max_link)

    #%% 统计每个数字出现的次数
    link_count = {}
    for i in links:
        link_count[i] = 0
    for traj in trajs:
        for link in traj:
            link_count[link] += 1

    #%%
    link_top = sorted(link_count.items(), key=lambda x: x[1], reverse=True)

    # # 只保留前100个
    # link_top = link_top[:100]

    #%%
    k = 100
    # link_topk = sorted(link_count.items(), key=lambda x: x[1], reverse=True)[:k]

    #%%
    import geopandas as gpd
    from shapely.geometry import LineString
    node_df = gpd.read_file('preprocessed_data/'+city+'_data/map/nodes.shp')
    edge_df = gpd.read_file('preprocessed_data/'+city+'_data/map/edges.shp')

    #%% 针对trajs构造training data，输入为trajs的起点和终点及link_top100中的link，输出为0，1标签，0表示traj不经过该link，1表示traj经过该link
    import random
    import torch

    data = []

    for traj in trajs:
        start = traj[0]
        end = traj[-1]
        # 查找start和end在edge_df中对应的经纬度
        start_coord = list(edge_df.iloc[start]['geometry'].coords)[0]
        end_coord = list(edge_df.iloc[end]['geometry'].coords)[0]
        min_longitude = min(start_coord[0], end_coord[0])
        max_longitude = max(start_coord[0], end_coord[0])
        min_latitude = min(start_coord[1], end_coord[1])
        max_latitude = max(start_coord[1], end_coord[1])
        # 筛选经纬度在start_coord和end_coord之内的link_top，直到达到100个
        link_topk = []
        count = 0
        for item in link_top:
            target_coord = list(edge_df.iloc[item[0]]['geometry'].coords)[0]
            if min_longitude <= target_coord[0] <= max_longitude and min_latitude <= target_coord[1] <= max_latitude:
                link_topk.append(item)
                count += 1
                if (count == k):
                    break
        while len(link_topk) < 100:
            link_topk.append(link_topk[-1])
        label = [0 for i in range(k)]
        for i, link in enumerate(link_topk):
            if link[0] in traj:
                label[i] = 1
        for i in range(k):
            data.append((start, end, link_topk[i][0], label[i]))

    #%% 检测data中每100个数据第四列包含1的频率
    cnt = 0
    for i in range(0, len(data), k):
        if sum([data[j][3] for j in range(i, i+k)]) > 0:
            cnt += 1
    print(cnt/len(trajs))

    #%% 检测data中第四列是1的频率
    cnt = 0
    for i in data:
        if i[3] == 1:
            cnt += 1
    print(cnt/len(data))

    #%%
    def get_batch(p1, p2):
        x_batch = np.zeros(((p2-p1), 3 * max_link))
        y_batch = np.zeros(((p2-p1),))
        z = 0
        for i in range(p1, p2):
            start_vec = np.zeros(max_link)
            end_vec = np.zeros(max_link)
            link_vec = np.zeros(max_link)
            start_vec[int(data[i][0])] = 1
            end_vec[int(data[i][1])] = 1
            link_vec[int(data[i][2])] = 1
            input = np.concatenate((start_vec, end_vec, link_vec))
            x_batch[z] = input
            y_batch[z] = data[i][3]
            z += 1
        return x_batch, y_batch

    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

    #%% 构造mlp分类器，输入为trajs的起点和终点及link_top100中的link，输出为0，1标签，0表示traj不经过该link，1表示traj经过该link
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from tqdm import tqdm

    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

        def predict(self, data, batch_size):
            score = 0
            for i in tqdm(range(0, len(data), batch_size)):
                batch_x, batch_y = get_batch(i, i + batch_size)
                batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
                batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
                y_pred = self.forward(batch_x)
                y_pred = torch.squeeze(y_pred, 1)
                for j in range(len(batch_y)):
                    if y_pred[j] > 0.5 and batch_y[j] == 1:
                        score += 1/k
                    elif y_pred[j] <= 0.5 and batch_y[j] == 0:
                        score += 1/k
            return score

        def predict_2(self, data, batch_size):
            score = 0
            score_2 = 0
            max_cnt = 0
            mid_point_list = []
            predict_true_list = []
            for i in tqdm(range(0, len(data), batch_size)):
            # for i in tqdm(range(0, 1000, batch_size)):
                batch_x, batch_y = get_batch(i, i + batch_size)
                batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
                batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
                y_pred = self.forward(batch_x)
                y_pred = torch.squeeze(y_pred, 1)
                j = torch.argmax(y_pred)
                mid_point_vec = batch_x[j][2*max_link:].cpu().numpy()
                # 找到1的位置作为mid_point
                mid_point = np.where(mid_point_vec == 1)[0][0]
                mid_point_list.append(mid_point)
                if batch_y[j] == 1:
                    score += 1
                    predict_true_list.append(1)
                else:
                    predict_true_list.append(0)
                # 如果batch_y中有1，max_cnt加1
                if sum(batch_y) > 0:
                    max_cnt += 1
                for j in range(len(batch_y)):
                    if y_pred[j] > 0.5 and batch_y[j] == 1:
                        score_2 += 1/k
                    elif y_pred[j] <= 0.5 and batch_y[j] == 0:
                        score_2 += 1/k
            return score, max_cnt, score_2, mid_point_list, predict_true_list
        #
        # def predict_proba(self, x):
        #     x = self.forward(x)
        #     return F.softmax(x, dim=1)
        #
        # def score(self, x, y):
        #     y_pred = self.predict(x)
        #     return torch.mean((y_pred == y).float())

        def fit(self, data, epochs, batch_size, lr, weight_decay):
            min_loss = 1000
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(epochs):
                for i in tqdm(range(0, len(data), batch_size)):
                    optimizer.zero_grad()
                    batch_x, batch_y = get_batch(i, i+batch_size)
                    batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
                    batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
                    y_pred = self.forward(batch_x)
                    y_pred = torch.squeeze(y_pred, 1)
                    loss = criterion(y_pred, batch_y)
                    # if loss.item() < min_loss:
                    #     min_loss = loss.item()
                    #     torch.save(self.state_dict(), 'new_code/param/mlp_'+str(trajs_num)+'_'+str(k)+'.pth')
                    loss.backward()
                    optimizer.step()
                print("epoch: {}, loss: {}".format(epoch, loss.item()))

    #%%
    batch_size_tmp = k

    #%%
    mlp = MLP(3 * max_link, 100, 1).to(device)
    mlp.fit(data, 10, batch_size_tmp, 0.01, 0.0001)
    torch.save(mlp.state_dict(), 'param/'+city+'_mlp_'+str(trajs_num)+'_'+str(k)+'.pth')

    #%% 读取模型参数
    mlp = MLP(3 * max_link, 100, 1).to(device)
    mlp.load_state_dict(torch.load('param/'+city+'_mlp_'+str(trajs_num)+'_'+str(k)+'.pth'))

    #%%
    print('start predict')
    score, max_cnt, score_2, mid_point_list = mlp.predict_2(data, batch_size_tmp)
    print(score, max_cnt, score/max_cnt, score_2)

    #%% 存储mid_point_list到txt
    with open('param/'+city+'_mid_point_list.txt', 'w') as f:
        for item in mid_point_list:
            f.write(str(item)+'\n')
        f.close()
