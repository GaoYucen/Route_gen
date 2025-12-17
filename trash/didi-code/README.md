# 代码逻辑

途经点路线生成思路

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

- CRP.py:

   - 生成6层分层路网，计算分区和边界节点

   - 用双向Dijkstra进行搜索，确定在边界节点上的候选点

   - 根据长度进行排序，并根据重叠率进行筛选，最终选出至多5个候选点

- candidate_CRP.py: 生成候选点和label信息
- via-node-predict.py: MLP预测模型，验证准确率和在轨迹上的概率
- compare_CRP.py: 验证途经点作为中间点生成路线的效果



结果：针对成都100条轨迹

找到途经点的比例：0.287

有候选点是途经点的路线比例：0.82

途经点最高F1-Score均值：0.72

Dijkstra F1-Score：0.64



via-node-predict预测准确率: 83.77%

预测出途经点在轨迹上的比例：0.6

途经点F1-Score：0.68

