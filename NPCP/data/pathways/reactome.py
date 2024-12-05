import re
import networkx as nx
import pandas as pd
from os.path import join
from config_path import REACTOM_PATHWAY_PATH
from data.gmt_reader import GMT

import matplotlib as plt

reactome_base_dir = REACTOM_PATHWAY_PATH
relations_file_name = 'ReactomePathwaysRelation.txt'
pathway_names = 'ReactomePathways.txt'
pathway_genes = 'ReactomePathways.gmt'


def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def complete_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph


def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))

    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n)
            next = net.successors(n)
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
        layers.append(dict)
    return layers


class Reactome():

    def __init__(self):
        self.pathway_names = self.load_names()   ## 获取当前各个通路的编号以及名字
        self.hierarchy = self.load_hierarchy()   ## 获取各个通路彼此之间的关系
        self.pathway_genes = self.load_genes()   ## 获取各个通路的名字、编号以及它内部所包含的基因！

    def load_names(self):    ## 导入通路
        filename = join(reactome_base_dir, pathway_names)             ### 目前来导入 ReactomePathways.txt 这个文件，此文件中一共有三列分别是 'reactome_id', 'pathway_name', 'species'  分别表示当前通路在此数据库以及文件中的编号，此通路的名字，其中species 表示当前通路属于具体哪个物种
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['reactome_id', 'pathway_name', 'species']
        return df

    def load_genes(self):
        filename = join(reactome_base_dir, pathway_genes)
        gmt = GMT()
        df = gmt.load_data(filename, pathway_col=1, genes_col=3)
        return df

    def load_hierarchy(self):    ## 导入层次结构
        filename = join(reactome_base_dir, relations_file_name)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['child', 'parent']
        return df


class ReactomeNetwork():

    def __init__(self):
        self.reactome = Reactome()  # low level access to reactome pathways and genes
        self.netx = self.get_reactome_networkx()

    def get_terminals(self):
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes

    def get_roots(self):

        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    # get a DiGraph representation of the Reactome hierarchy           获得Reactome层次结构的DiGraph表示法
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):       ## 判断对象是否包含某属性（即当前RN类是否包含netx这个属性）
            return self.netx
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only    目前是做前列腺癌症的，因此在这里需要只用人的通路信息！
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
        print("目前是在data/pathways/reactome.py这个文件中，大梦谁先绝！此时测试一下构建的这个网络情况是怎样的！", net)
        net.name = 'reactome'

        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]     ### 获取入度为零的哪些点（就是第一层的节点）
        root_node = 'root'
        edges = [(root_node, n) for n in roots]           ### 根节点与第一层中的各个节点依次相连，构造对应的边
        net.add_edges_from(edges)

        print("目前是在data/pathways/reactome.py这个文件中，最终构造的这个网络情况是怎样的！", net)

        # plt.figure(figsize=(8, 8))
        # nx.draw(net, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
        # plt.axis('equal')
        # plt.show()

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):

        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')     ### 广度优先搜索构造的树

        return G

    def get_completed_network(self, n_levels):
        G = complete_network(self.netx, n_leveles=n_levels)
        return G

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_leveles=n_levels)
        return G

    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        # get the last layer (genes level)
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]  # set of terminal pathways
        # we need to find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            genes = genes_df[genes_df['group'] == pathway_name]['gene'].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes


        # print("当前是在data/pathways/reactome.py 文件中，当前所获取的这个字典的情况是怎样的！", dict)

        layers.append(dict)
        return layers
