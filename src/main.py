import numpy as np
import pandas pd
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import time


from network import AntaiRSModel
import config as cfg
from dataset import data_norm,AntaiDataset
from utils import create_folder, to_cuda_if_available,SaveBest,EarlyStopping

class AntaiRSModel(nn.Module):
    def __init__(self):
        super(AntaiRSModel,self).__init__()
        torch.manual_seed(1) # 随机初始化embedding举证
        self.admin_id_embeds = nn.Embedding(admin_id_max, cfg.EMBED_DIM) # 64
        self.item_id_embeds = nn.Embedding(item_id_max, cfg.EMBED_DIM)
        self.item_sellnum_embeds = nn.Embedding(item_sellnum_max, cfg.EMBED_DIM)
        self.cate_id_embeds = nn.Embedding(cate_id_max, cfg.EMBED_DIM//2)
        self.store_id_embeds = nn.Embedding(store_id_max, cfg.EMBED_DIM//2)
        self.item_price_embeds = nn.Embedding(item_price_max, cfg.EMBED_DIM)
        # 用户dense 初始化
        self.uid_fc = nn.Linear(cfg.EMBED_DIM, cfg.FC1_DIM)
        self.admin_fc1 = nn.Linear(cfg.ADMIN_FEATURE_DIM,cfg.FC1_DIM)
        self.admin_fc2 = nn.Linear(cfg.FC1_DIM*2,cfg.FC2_DIM)
        # 商品dense 初始化
        self.itemid_fc = nn.Linear(cfg.EMBED_DIM, cfg.FC1_DIM)
        self.sellnum_fc = nn.Linear(cfg.EMBED_DIM, cfg.FC1_DIM)
        self.cateid_fc = nn.Linear(cfg.EMBED_DIM//2, cfg.FC1_DIM//2)
        self.storeid_fc = nn.Linear(cfg.EMBED_DIM//2, cfg.FC1_DIM//2)
        self.itemprice = nn.Linear(cfg.EMBED_DIM, cfg.FC1_DIM)
        self.item_fc = nn.Linear(cfg.FC1_DIM*4, cfg.FC2_DIM)
        
    def forward(self,admin, item): # bs x 14, bs x5
        # 用户信息embedding
        uid_embed_layer = self.admin_id_embeds(admin[:,0].long()) # bs x 1x embed_dim => bs x 64
        # 用户dense
        uid_dense = self.uid_fc(uid_embed_layer) # bs  x128
        admin_dense = self.admin_fc1(admin[:,1:])  # bs x 128
        # 用户concat + dense
        admin_concat = torch.cat((uid_dense, admin_dense),dim=1) # bsx 256
        admin_concat_out = self.admin_fc2(admin_concat)  # bsx 256
        # 商品信息embedding
        itemid_embed_layer = self.item_id_embeds(item[:,0].long())  # bsx64
        sellnum_embed_layer = self.item_sellnum_embeds(item[:,1].long()) # bs x64
        cateid_embed_layer = self.cate_id_embeds(item[:,2].long()) # bs x32
        storeid_embed_layer = self.store_id_embeds(item[:,3].long()) # bs x32
        itemprice_embed_layer = self.item_price_embeds(item[:,4].long())
        # 商品dense
        itemid_dense = self.itemid_fc(itemid_embed_layer) # bsx128
        sellnum_dense = self.sellnum_fc(sellnum_embed_layer) # bsx  128
        cateid_dense = self.cateid_fc(cateid_embed_layer) # bs x64
        storeid_dense = self.storeid_fc(storeid_embed_layer)  # bs x 64
        itemprice_dense = self.itemprice(itemprice_embed_layer)  # bs  x128
        # 商品concat + dense
        item_concat = torch.cat((itemid_dense,sellnum_dense,cateid_dense,storeid_dense, itemprice_dense),dim=1)  # bs x 128*4
        item_concat_out = self.item_fc(item_concat)  # bs x 256
        out = torch.sigmoid(torch.sum(admin_concat_out*item_concat_out,1))    # bs x 1    
        return out

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate','-lr', type=float, default=1e-3)
    parser.add_argument('--L2', '-l2', type=float, default=0)
    parser.add_argument('--model-name', '-m', type=str, default='test')
    parser.add_argument('--batch-size', '-bs', type=int, default=32)
    parser.add_argument('--num-earlystop', '-ne', type=int, default=50)
    parser.add_argument('--niter', '-n', type=int, default=300)
    global args
    args = vars(parser.parse_args())
    
    localtime_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    if args['model_name'] == 'test':
        dirname = args['model_name']
        store_dir = os.path.join("stored_data",dirname)
    else:
        dirname = "%s_bs%s_lr%s_" % (args['model_name'], args['batch_size'], args['learning_rate'])
        store_dir = os.path.join("stored_data", dirname + localtime_str)
    saved_model_dir = os.path.join(store_dir, "model")
    create_folder(store_dir)
    create_folder(saved_model_dir)
    
    print('start train...')
    print('read tranning data....')
    with open(cfg.TRAIN_PATH,'rb') as f:
        model_train_df = pickle.load(f)
    with open(cfg.TEST_PATH,'rb') as f:
        model_test_df = pickle.load(f)
    print('normalize data....')
    model_train_df  = data_norm(model_train_df)
    model_test_df = data_norm(model_test_df)
    print('init model parameters...')
    # 用户ID数
    admin_id_max = model_train_df['buyer_admin_id'].max()
    # 商品ID数
    item_id_max =  model_train_df['item_id'].max()
    # 商品卖出数
    item_sellnum_max =  model_train_df['num_sell'].max()
    # 商品种类数目
    cate_id_max  = model_train_df['cate_id'].max()
    # 商店种类数
    store_id_max = model_train_df['store_id'].max()
    #商品价格
    item_price_max = model_train_df['item_price'].max()
    
    print('init dataset...')
    anti_trian_dataset = AntaiDataset(model_train_df)
    antai_test_dataset = AntaiDataset(model_test_df)
    train_dataloader = DataLoader(anti_trian_dataset,batch_size=args['batch_size'], shuffle=True,num_workers=16)
    test_dataloader = DataLoader(antai_test_dataset,batch_size=args['batch_size'], shuffle=False,num_workers=16)
    print('init model....')
    model = AntaiRSModel()
    optim_kwargs = {"lr": args['learning_rate'],'eps':1e-8}
    print("optim_kwargs %s"%(optim_kwargs))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs)
    loss_fn = nn.BCELoss()
    
    early_stop = EarlyStopping(model, args['num_earlystop'], "sup")
    save_best = SaveBest("sup")
    [model,loss_fn] = to_cuda_if_available([model, loss_fn])
    for epoch in range(args['niter']):
        model.train()
        for i, batch_input in enumerate(data_loader):
            optimizer.zero_grad()
            [admin, item, label] = to_cuda_if_available([batch_input[0],batch_input[1]，batch_input[2]])
            pred = model(admin, item)
            loss = loss_fn(pred, target)
            
            
            
    
    
    
    
    