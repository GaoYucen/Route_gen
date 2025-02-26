# Route_gen

Environment:
- Python 3.11
- Pytorch 2.6.0

## Structure

###### data
The data can be downloaded at: [https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view](https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view)
- graph_sc.pkl：强连通图
- [train, valid, test]_data_small_sc：训练，测试，验证集的小规模集合，数据格式是点序列 <br>
  格式为：[['20161108', 180561, 1478601188, 1478601744], [3562055474, 7877083919, 7877083908,...]]，分别代表日期，id，开始时间，结束时间，点序列

###### preprocessed
- node2vec_sc.emb：Node2Vec的参数
- node_embedding_sc.pkl：Node2Vec学到的点嵌入
- partitions.pkl：CRP分层结果
- boundary_nodes.pkl：CRP分层的边界节点
- [train, valid, test]_data_samples.pkl：训练，测试，验证集的轨迹数据
- [train, valid, test]_candidate_list.pkl：训练，测试，验证集的途经点候选集
- [train, valid, test]_on_traj_flag_list.pkl：训练，测试，验证集的途经点是否在轨迹上的标记 label
- test_selected_points.pkl：测试集的预测点

###### code：
- preprocess: 数据预处理（我们假设以节点node为关注点）：
   - Node2Vec.py：获取节点的嵌入，128维
- config：hyper-parameter
- CRP
  - CRP_partition.py: CRP分层
  - candidate_CRP.py: CRP分层生成途经点label
- via_node_predict_regular_sample.py: 预测途经点的MLP模型并验证效果（正则化）
  - 提高sample_acc：修改loss
  - padding + batch: 保持sample计算情况下加速运算
  - 提升泛化性
    - 增强正则化
      - 提高 Dropout 率到 0.3 
      - 在每个主要层后添加 LayerNorm 
      - 增加 L2 正则化 (weight decay)
      - 添加梯度裁剪 
    - 改进模型架构 
      - 增加网络深度   
      - 添加残差连接 
      - 扩大中间层维度 
    - 优化训练策略 
      - 使用 AdamW 优化器
      - 实现余弦退火学习率调度 
      - 添加早停机制 
      - 保存最佳模型 
    - 添加监控 
      - 记录训练和验证的损失与准确率 
      - 打印详细的训练信息
- valid: 验证途经点效果
   - compare_CRP.py：CRP途经点结果 vs 直接Dijkstra结果

## To run:

### data preparation
```
cd Route_gen
python code/preprocess/data_preprocess_for_map.py
python code/preprocess/Node2Vec.py
python code/CRP/CRP.py
python code/CRP/candidate_CRP.py
```

### analyze the dataset
```
python code/analyze/train_val_test_analysis.py
```

### train via-node prediction model
```
python code/via_node_predict_regular_sample.py
```

### test via-node prediction model
```
python code/valid/compare_CRP.py
```

## Experimental results

- candidate_CRP参数测试：
  - 基础结果
    - 找到途经点的比例：0.287
    - 有候选点是途经点的路线比例：0.82
    - 途经点最高F1-Score均值：0.72
    - Dijkstra F1-Score：0.64
  - ==参数测试==
    - ==num_routes（初始5，调整8）增加候选点数量后，0.24；0.86，会降低比例，但增大路线比例==
    - ==theta（初始0.75，调整0.6）降低减少因路线重叠导致的筛选后，0.284；0.79，会增大比例，但降低路线比例，推测原因为路线越重叠，对于不好找的路找到的概率越低。加大阈值到0.9后，变为0.3,0.85，加到到0.95后，变为0.3,0.84，没有明显提升==
      
  - 基于最新参数
    - F1 Score mean: 0.826834843161527
    - Dijkstra F1 Score mean: 0.642584758586133
    - F1 Score on mean: 0.8796115352782202
    - Dijkstra F1 Score on mean: 0.6726866747380136
