train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
val_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
train_set = myFloder(train_root, load_graphs)
test_set = myFloder(test_root, load_graphs)
if opt.val:
    val_set = myFloder(val_root, load_graphs)

print('train number:', train_set.size)
print('test number:', test_set.size)
print('user number:', user_num)
print('item number:', item_num)
f = open(opt.data+'_neg', 'rb')
data_neg = pickle.load(f) # 用于评估测试集
train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=12)
test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=8)
if opt.val:
    val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)

# 初始化模型
model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
             user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short,
             item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item,
             layer_num=opt.layer_num).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.CrossEntropyLoss()
best_result = [0, 0, 0, 0, 0, 0]   # hit5,hit10,hit20,mrr5,mrr10,mrr20
best_epoch = [0, 0, 0, 0, 0, 0]
stop_num = 0
