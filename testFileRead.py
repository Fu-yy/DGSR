import os

import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
import networkx as nx

import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed

def getFile():
    # 获取当前文件的目录
    current_dir = os.path.dirname(__file__)


    # 拼接文件路径
    file_path = os.path.join(current_dir, 'Data', 'test.txt')
    # 将斜杠替换为反斜杠
    file_path = file_path.replace('/', '\\')

    print(file_path)

def generate_user():
    return 1,2

def generate_data():
    user = [1,2]
    a = Parallel(n_jobs=-1)(delayed(lambda u: generate_user())(u) for u in user)
    return a

if __name__ == '__main__':
    # 加载图列表
    graphs, _ = dgl.load_graphs("E:\MyCode\PycharmCode\DGSR\Data\Games_graph.bin")
    graphs1, _ = dgl.load_graphs("E:\MyCode\PycharmCode\DGSR\Data\Beauty_graph.bin")
    graph = graphs[0]  # 提取第一个图
    graph1 = graphs1[0]  # 提取第一个图

    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Number of edges:", graph1.number_of_nodes())
    print("Number of edges:", graph1.number_of_edges())
    print("Number of edges:", type(graph))

