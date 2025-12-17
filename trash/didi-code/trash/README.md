# 代码逻辑

途经点路线生成思路示意图

![示意图](/Users/gaoyucen/Library/Mobile Documents/com~apple~CloudDocs/（21-11-2）论文/会议4. IJCAI-滴滴轨迹生成/figure/示意图.png)

因为实走轨迹并非最短路，我们期望找到实走轨迹上的点作为中间点，通过【起点到中间点】【中间点到终点】的两段路线拼接生成方式，提高和实走轨迹的相似度



Environment:

- Python 3.11



data:

使用2025010312的的路网，1月1日的实走轨迹

边数据：

[hdfs://DClusterNmg3/](hdfs://DClusterNmg3/)user/bigdata-dp/gfeo/link_weight_opt/nn/source_data/pipeline_data/2025010312_1_link_feat/*

边映射关系表：

[hdfs://DClusterNmg](hdfs://DClusterNmg)3/user/bigdata-dp/gfeo/link_weight_opt/nn/source_data/pipeline_data/2025010312_1_link_idx/* 格式为eswlinkid:link_idx

2025年1月1日实走轨迹：[hdfs://DClusterNmg3/user/bigdata-dp/multi_route/online_trip_often_route_db/route_corpus_city_mining/map_version=2025010312/clean_routes_allcities/year=2025/month=01/day=01/city_id=1/part-06433-a4421d32-9bc3-4a0e-9e26-a91274e43f00.c000.txt](hdfs://DClusterNmg3/user/bigdata-dp/multi_route/online_trip_often_route_db/route_corpus_city_mining/map_version=2025010312/clean_routes_allcities/year=2025/month=01/day=01/city_id=1/part-06433-a4421d32-9bc3-4a0e-9e26-a91274e43f00.c000.txt)

- link_feature: 边信息，linkid从0开始排序
- link_map: 边映射信息，格式为esiweiid: linkid
- traj.txt: 序列信息
- link_new.csv: 北京边数据，
- traj_small.txt: 只保留边序列的1000条轨迹
- traj_osmid_small.txt: 边在北京路网中，且边转化为点的993条点序列轨迹



code：

- road_new.py: 针对link_feature和link_map进行处理，得到link_new.csv，格式为[esiweiid, snode, enode, dist/length]四维特征，得到的networkx路网包括1,208,474个节点，2,775,761条边

- preprocess.py: 

   - 仅保留轨迹的边序列信息，存储到traj_small.txt中
   - 将边序列转化成点序列，存储到traj_osmid_small.txt中
   - 检查原轨迹是否是连续的→连续

- preprocess_traj.py: 在traj_osmid_small的轨迹中能找到从起点到终点最短路的215条轨迹

- bidijk.py:

   - 使用Dijkstra生成路线

   - 使用实走轨迹的中点作为中间点，再用Dijkstra生成两段路线，拼接成完整的路线

   - 使用双向Dijkstra确定前k个碰撞点（暂时k取3，且要求由碰撞点生成的两段路线的总长度在最短路总长度的(1.1, 1.5]之间，按道路长度从小到大排序），检验使用碰撞点作为中间点，分成两段生成路线的效果（这部分代码可能还需要检验）



res: 因为Dijkstra跑的比较慢，故先在小规模数据上做实验

- 针对100条轨迹

|                    | Similarity（节点重复平均数） | Precision | Recall | F1-Score |
| ------------------ | ---------------------------- | --------- | ------ | -------- |
| Dijkstra           | 94.8                         | 0.711     | 0.681  | 0.694    |
| 以轨迹中点为中间点 | 116.77                       | 0.858     | 0.836  | 0.846    |

- 针对10条轨迹

|                    | Similarity（节点重复平均数） | Precision | Recall | F1-Score |
| ------------------ | ---------------------------- | --------- | ------ | -------- |
| Dijkstra           | 84.7                         | 0.654     | 0.624  | 0.636    |
| 以轨迹中点为中间点 | 99.5                         | 0.810     | 0.805  | 0.807    |
| 以碰撞点1为中间点  | 70.1                         | 0.509     | 0.580  | 0.540    |
| 以碰撞点2为中间点  | 70.6                         | 0.507     | 0.578  | 0.538    |
| 以碰撞点3为中间点  | 71.4                         | 0.554     | 0.618  | 0.581    |
