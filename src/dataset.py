from torch.utils.data import Dataset
import numpy as np


def data_norm(data_df):
    # 先整理列名方便后面检索，然后进行数据的归一化
    data_df = data_df[['buyer_admin_id',
                       'num_shop',  #购买次数，
                       'num_item',  #购买商品件数，
                       'pop_unpop_rate',  #热销商品与非热销商品件数的比重，
                       'item_price_sum',  # 商品总价格
                       'item_price_max',  #商品最大价格
                       'item_price_min',  #商品最小价格
                       'item_price_mean',  #商品价格均值
                       'item_price_median', #商品价格中值
                       'num_item_cate',  #商品类别数
                       'num_store_cate', #商店类别数
                       'shoptime_1',  #在0-8时间段购买商品数所占比重
                       'shoptime_2', #在8-16时间段...
                       'shoptime_3',  #在16-24时间段..
                       'item_id',
                       'num_sell',
                       'cate_id',
                       'store_id',
                       'item_price',
                       'label'
                        ]]
    data_df.iloc[:,1:14] = data_df.iloc[:,1:14].apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x))) # 数据相关归一化
    return data_df


class AntaiDataset(Dataset):
    def __init__(self,data_df):
        self.data_df = data_df
        
    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, index):
        data_series = self.data_df.iloc[index]
        label = data_series['label']
        admin_features = data_series.iloc[:14].values
        item_features = data_series.iloc[14:-1].values
        return admin_features.astype(np.float32), item_features.astype(np.float32), np.array(label, dtype=np.float32)