import torch 
import torch.nn as nn
import config as cfgI

class AntaiRSModel(nn.Module):
    def __init__(self):
        super(AntaiRSModel,self).__init__()
        torch.manual_seed(1) # 随机初始化embedding矩阵
        self.admin_id_embeds = nn.Embedding(admin_id_max, embed_dim) # 64
        self.item_id_embeds = nn.Embedding(item_id_max, embed_dim)
        self.item_sellnum_embeds = nn.Embedding(item_sellnum_max, embed_dim)
        self.cate_id_embeds = nn.Embedding(cate_id_max, embed_dim//2)
        self.store_id_embeds = nn.Embedding(store_id_max, embed_dim//2)
        self.item_price_embeds = nn.Embedding(item_price_max, embed_dim)
        # 用户dense 初始化
        self.uid_fc = nn.Linear(embed_dim, fc1_dim)
        self.admin_fc1 = nn.Linear(admin_feature_dim,fc1_dim)
        self.admin_fc2 = nn.Linear(fc1_dim*2,fc2_dim)
        # 商品dense 初始化
        self.itemid_fc = nn.Linear(embed_dim, fc1_dim)
        self.sellnum_fc = nn.Linear(embed_dim, fc1_dim)
        self.cateid_fc = nn.Linear(embed_dim//2, fc1_dim//2)
        self.storeid_fc = nn.Linear(embed_dim//2, fc1_dim//2)
        self.itemprice = nn.Linear(embed_dim, fc1_dim)
        self.item_fc = nn.Linear(fc1_dim*4, fc2_dim)
        
    def forward(self,admin, item): # bs x 14, bs x5
        # 用户信息embedding
#         admin = admin.float()
#         item = item.float()
#         ipdb.set_trace()
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