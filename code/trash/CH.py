import networkx as nx


def crp_node_hierarchy(G):
    """
    根据 CRP 算法思想，基于节点度对有向图节点进行分层
    :param G: 有向图对象
    :return: 一个字典，键为层数，值为该层的节点列表
    """
    # 计算每个节点的度（入度 + 出度）
    node_degrees = {node: G.in_degree(node) + G.out_degree(node) for node in G.nodes()}
    # 按照节点度从大到小排序
    sorted_nodes = sorted(node_degrees.items(), key=lambda item: item[1], reverse=True)

    hierarchy = {}
    current_level = 0
    current_degree = None

    for node, degree in sorted_nodes:
        if current_degree is None or degree != current_degree:
            # 如果是新的度值，进入新的层级
            current_level += 1
            current_degree = degree
            hierarchy[current_level] = []
        # 将节点添加到当前层级
        hierarchy[current_level].append(node)

    return hierarchy


# 示例使用
import pickle

with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()

# 调用函数获取节点分层信息
node_hierarchy = crp_node_hierarchy(G)

# 输出每层的节点信息
for level, nodes in node_hierarchy.items():
    print(f"Level {level}: {nodes}")