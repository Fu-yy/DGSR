
if __name__ == "__main__":
    # 设置随机种子
    seed(430)  # 设置随机种子为430，用于实现可复现的随机性
    random.seed(430)
    torch.manual_seed(430)
    torch.cuda.manual_seed_all(430)
    torch.backends.cudnn.deterministic = True

    SZ = 12  # 样本大小
    SEQ_LEN = 10  # 序列长度

    # 对数据集进行采样
    sample_relations(conf['dataset.name'], conf['dataset.n_items'], sample_size=SZ)  # 根据指定的数据集名称和物品数量，采样样本大小为SZ

    # 对图进行操作，获取图g和物品数量item_num
    g, item_num = uui_graph(conf['dataset.name'], sample_size=SZ, topK=20, add_u=False, add_v=False)
    # 通过使用指定的数据集名称和采样大小SZ，从图中获取图g和物品数量item_num
    # topK表示保留每个节点的最高K个邻居节点，add_u和add_v表示是否添加边为(u, v)的逆边和(v, u)的逆边

    print(g)  # 打印图g

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断可用设备，如果有CUDA则使用GPU，否则使用CPU

    train_data = SessionDataset(train_data, conf, max_len=SEQ_LEN)
    # 创建训练数据集，使用train_data和配置conf，设置最大序列长度为10
    # train_data表示训练数据，conf表示配置信息，max_len表示序列的最大长度

    test_data = SessionDataset(test_data, conf, max_len=SEQ_LEN)
    # 创建测试数据集，使用test_data和配置conf，设置最大序列长度为10
    # test_data表示测试数据，conf表示配置信息，max_len表示序列的最大长度

    train_iter = DataLoader(dataset=train_data,
                            batch_size=conf["batch_size"],
                            num_workers=4,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=False)
    # 创建训练数据迭代器
    # dataset表示要迭代的数据集，batch_size表示每个批次的样本数量，num_workers表示用于数据加载的子进程数量
    # drop_last表示是否丢弃最后一个批次（如果批次大小不完整），shuffle表示是否对数据进行随机洗牌，pin_memory表示是否将数据加载到固定内存中

    test_iter = DataLoader(dataset=test_data,
                           batch_size=conf["batch_size"] * 16,
                           num_workers=4,
                           drop_last=False,
                           shuffle=False,
                           pin_memory=False)
    # 创建测试数据迭代器
    # dataset表示要迭代的数据集，batch_size表示每个批次的样本数量，num_workers表示用于数据加载的子进程数量
    # drop_last表示是否丢弃最后一个批次（如果批次大小不完整），shuffle表示是否对数据进行随机洗牌，pin_memory表示是否将数据加载到固定内存中

    model = HG_GNN(g, conf, item_num, SEQ_LEN).to(device)
    # 创建模型实例
    # g表示图数据，conf表示配置信息，item_num表示物品数量，SEQ_LEN表示序列长度
    # 将模型移动到设备(device)，使用GPU加速计算（如果可用）

    train(conf, model, device, train_iter, test_iter)
    # 进行训练
    # conf表示配置信息，model表示模型，device表示设备，train_iter表示训练数据迭代器，test_iter表示测试数据迭代器