# 代码逻辑

Environment:

- Python 3.11
- Pytorch 2.2.1

The data can be downloaded at: [https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view](https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view)



data描述

![截屏2025-02-13 01.22.18](/Users/gaoyucen/Desktop/截屏2025-02-13 01.22.18.png)



data 以chengdu_data为例：

- graph_sc.pkl：强连通图
- node_nbrs_sc.pkl：节点的后续节点
- node2vec_sc.emb：Node2Vec的参数
- node_embedding_sc.pkl：Node2Vec学到的点嵌入
- preprocessed-[train, test, validation]-trips-[all, small]：训练，测试，验证集的全集和小规模集合，边是以0开始的边序列编号

![Image.png](https://res.craft.do/user/full/8fb2e49f-91ba-7fe3-46a4-c9eaf024703a/doc/7A88DF1E-035F-4A90-8C07-B91210C6239C/E5A6C3FF-F32E-4546-9592-84C60E3DDCEF_2/6L35xOcTJqvLlUwMLXebpEVYujZ7gprzNmOhu0yTvuUz/Image.png)

   规模分别为1,448,940；1,954,551；366,617

- preprocessed-[train, test, validation]-trips-small-osmid：点序列，点是osmid，规模是10k条
- [train, test, validation]_data_small_sc.pkl：从small-osmid中提取的点都在graph_sc中的轨迹
- map：点和边信息
   - graph.pkl和graph_with_haversine.pkl是原作者提供的pkl数据，目前shapely>2.0的版本已不太支持，尽量不使用

code：

- config：hyper-parameter
- preprocess: 数据预处理（我们假设以节点node为关注点）：
   - data_preprocess_for_traj.py：处理轨迹数据，得到以点为序列的轨迹，格式形如[ [data, orderid, timestamp_start, timestamp_end], [点序列] ]
   - data_preprocess_for_scc.py：将graph变成强连通图graph_scc，包含3882nodes，9058edges，并将数据集中不在graph_scc里的轨迹删掉得到scc数据，规模分别为9431；9442；9426
   - data_preprocess_for_map.py：计算后续节点
   - Node2Vec.py：获取节点的嵌入，64维
- model: 机器学习模型学习与验证
   - model：定义模型结构
   - train：训练机器学习模型实现路线生成
      - 对于我们的场景，不需要scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5
      - 后续可以验证下加入softmax的效果
   - valid：验证路线生成的效果

![Image.png](https://res.craft.do/user/full/8fb2e49f-91ba-7fe3-46a4-c9eaf024703a/doc/7A88DF1E-035F-4A90-8C07-B91210C6239C/ABE11412-BC2A-4830-A64F-A502C08D6EA4_2/y1LF37cBAZzTPwLsBIcVfoxyvN65TgCn7BRLBv0pgd4z/Image.png)

![Image.png](https://res.craft.do/user/full/8fb2e49f-91ba-7fe3-46a4-c9eaf024703a/doc/7A88DF1E-035F-4A90-8C07-B91210C6239C/92CB28AD-D171-4F8F-9610-7614FD3D383C_2/dUxxhqj6AZcEsIHRCN82rGkTtYnyxPoL7EoL60TOpOYz/Image.png)

- valid: 验证途经点效果
   - compare_CRP.py：对比CRP结果

![Image.png](https://res.craft.do/user/full/8fb2e49f-91ba-7fe3-46a4-c9eaf024703a/doc/7A88DF1E-035F-4A90-8C07-B91210C6239C/484CB302-2BCD-4FB7-86E0-9DFEF20DA8BA_2/qaSrhNIlQOLDrQyP8T8t5EMGs0SBIfy1ydUtydU34i8z/Image.png)

   - compare_dijkstra: 使用Dijkstra's algorithm作为路线生成方法，验证加中间点生成的效果

![Image.png](https://res.craft.do/user/full/8fb2e49f-91ba-7fe3-46a4-c9eaf024703a/doc/7A88DF1E-035F-4A90-8C07-B91210C6239C/EA730A60-1FCE-4B55-95FD-D780260974C8_2/bx2cJqAYLZjEaZbpUyHwb1bJSkwMy3W7gtEaiaVGPP8z/Image.png)

   - valid_waypoint：将轨迹分为起点→中点，中点→终点两段，当两段都抵达时，算作整段路线抵达，验证路线生成效果

![Image.png](https://res.craft.do/user/full/8fb2e49f-91ba-7fe3-46a4-c9eaf024703a/doc/7A88DF1E-035F-4A90-8C07-B91210C6239C/5C218C15-52CF-4F13-9930-BF4321E1DD6A_2/JvqyHGZyfbiPyEMX8KxCFZYXLGCZkye2ZcANwaLKMJ8z/Image.png)

![Image.png](https://res.craft.do/user/full/8fb2e49f-91ba-7fe3-46a4-c9eaf024703a/doc/7A88DF1E-035F-4A90-8C07-B91210C6239C/FC6C286B-273D-4A18-90AA-907FD1F933EE_2/6OdKHCnNfXRFxqOvYO8yXM9xx958AFxdkDxazt2DNiQz/Image.png)

   - waypoint_dijkstra: 生成dijkstra的中点
   - valid_waypoint_dijkstra: 以Dijkstra的中点为中点，以learning模型生成路线

![Image.png](https://res.craft.do/user/full/8fb2e49f-91ba-7fe3-46a4-c9eaf024703a/doc/7A88DF1E-035F-4A90-8C07-B91210C6239C/0D79E464-56F6-41CB-809C-DF9096A6EF79_2/QVsGoWC13kQ4stB87u1YLmlKvINmMPnds6RxkjCVvAIz/Image.png)

- CRP
   - CRP: CRP分层
   - candidate_CRP: CRP分层生成途经点label
      - 基础结果
        - 找到途经点的比例：0.287
        - 有候选点是途经点的路线比例：0.82
        - 途经点最高F1-Score均值：0.72
        - Dijkstra F1-Score：0.64
      - ==参数测试==
        - ==num_routes（初始5，调整8）增加候选点数量后，0.24；0.86，会降低比例，但增大路线比例==
        - ==theta（初始0.75，调整0.6）降低减少因路线重叠导致的筛选后，0.284；0.79，会增大比例，但降低路线比例，推测原因为路线越重叠，对于不好找的路找到的概率越低。加大阈值到0.9后，变为0.3,0.85，加到到0.95后，变为0.3,0.84，没有明显提升==
      
      - 基于最新参数
      
        F1 Score mean: 0.826834843161527
        Dijkstra F1 Score mean: 0.642584758586133
        F1 Score on mean: 0.8796115352782202
        Dijkstra F1 Score on mean: 0.6726866747380136
   
- via-node-predict.py: 预测途经点的MLP模型并验证效果

   性能：

   - 准确率：83.77%
   - 预测出途经点在轨迹上的比例：0.6

- ==Via-node-predict_2.py：修订版本==

   - ==Selected points ratio: 0.94==
   - ==CRP平均f1-score: 0.7675693578734871==

   

目前的核心问题：

1. 分成两段用learning方法生成路线，会导致抵达率变低→可以修改抵达逻辑，对于中点未抵达的情况仍然向后运行





修改方向：

https://www.doubao.com/thread/w1ef3698fcc61d947



问题：

1. 需要修订acc和selected ratio差异的问题

2. 如何能增强训练效果

![截屏2025-02-13 04.00.22](/Users/gaoyucen/Desktop/截屏2025-02-13 04.00.22.png)

![截屏2025-02-13 04.01.00](/Users/gaoyucen/Library/Application Support/typora-user-images/截屏2025-02-13 04.01.00.png)

3. softmax和crossentropyloss的配合：使用crossentropyloss时要去掉softmax





代码运行顺序：

1. Data_preprocess_for_traj_small获取osmid数据
2. Data_preprocess_for_scc获取强连通图


## data preparation
```
cd Route_gen
python code/preprocess/data_preprocess_for_traj_small.py --chengdu_data
python code/preprocess/data_preprocess_for_scc.py --chengdu_data
python code/preprocess/Node2Vec.py --chengdu_data
python code/preprocess/data_preprocess_for_map.py --chengdu_data
python code/CRP/CRP.py --chengdu_data
python code/CRP/candidate_CRP.py --chengdu_data
```

## train next-node prediction model
```
python code/model/train.py
```

## train via-node prediction model

