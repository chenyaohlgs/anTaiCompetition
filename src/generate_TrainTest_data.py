import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import ipdb

def generate_data(data, admin_features,item_features):
    admin_ids = admin_features['buyer_admin_id'].to_list()
    all_data = []
    missing_count = 0
    for admin_id in tqdm(admin_ids):
        item_ids = data[data['buyer_admin_id']==admin_id]['item_id'].to_list()
        tmp_data = []
        try:
            for item_id in item_ids:
                tmp_data.append(np.concatenate([admin_features[admin_features['buyer_admin_id'] == admin_id].values[0],
                                                item_features[item_features['item_id'] == item_id].values[0],
                                                np.array([1])]))
            all_data.extend(tmp_data)
        except:
            missing_count +=1
            pass
    print('missing item num: %s'%missing_count)
    return np.array(all_data)

def drop_lose_item(data_df,item_df, lose_item):
    length = len(list(lose_item))
    all_drop_index= []
    for i,item in enumerate(list(lose_item)):
        drop_index = data_df[data_df['item_id'] == item].index[:]
        all_drop_index.extend(list(drop_index.values))
    data_df_filter = data_df.drop(all_drop_index).reset_index(drop=True)  # 去掉item不在商品表中的数据
    data_df_filter = pd.merge(data_df_filter,item_df, how='left',on='item_id') # 将商品表中的数据添加到原始dataframe中   
    return data_df_filter

def merge_data(data_df, admin_features, item_features):
    tmp_df1 = pd.merge(data_df[['buyer_admin_id','item_id']],admin_features,how='outer',on=['buyer_admin_id'])
    tmp_df2 = pd.merge(tmp_df1, item_features,on=['item_id'],how='outer')
    tmp_df2['label'] = 1
    return tmp_df2
    

if __name__=='__main__':
    TRAING_ADMIN_PATH = 'features/train_admin_features.pkl'
    TRAING_ITEM_PATH = 'features/train_item_features.pkl'
    TEST_ADMIN_PATH = 'features/test_admin_features.pkl'
    TEST_ITEM_PATH = 'features/test_item_features.pkl'
    TRAING_DATA_PATH = '../data/train.pkl'  # 读取原始训练集
    TEST_DATA_PATH = '../data/test.pkl'
    ITEMATTR_PKL_PATH = '../data/item_attr.pkl'

    with open(TRAING_ADMIN_PATH,'rb') as f:
        train_admin_features = pickle.load(f)
    with open(TRAING_ITEM_PATH,'rb') as f:
        train_item_features = pickle.load(f)
    with open(TEST_ADMIN_PATH,'rb') as f:
        test_admin_features = pickle.load(f)
    with open(TEST_ITEM_PATH,'rb') as f:
        test_item_features = pickle.load(f)
    with open(TRAING_DATA_PATH,'rb') as f:
        train_df = pickle.load(f)
    with open(TEST_DATA_PATH,'rb') as f:
        test_df = pickle.load(f)
    with open(ITEMATTR_PKL_PATH,'rb') as f:
        item_df = pickle.load(open(ITEMATTR_PKL_PATH, 'rb'), encoding='iso-8859-1')
    
    print('处理缺失值.....')
    train_lose_item = set(train_df['item_id'].unique()).difference(set(item_df['item_id'].unique()))
    test_lose_item = set(test_df['item_id'].unique()).difference(set(item_df['item_id'].unique()))
    print('drop lose item .....')
    train_filter = drop_lose_item(train_df, item_df, train_lose_item)
    test_filter = drop_lose_item(test_df, item_df, test_lose_item)
    print('generate train data....')
    model_train_data = merge_data(train_filter, train_admin_features, train_item_features)
    print('generate test data....')
    model_test_data = merge_data(test_filter, test_admin_features,test_item_features)
    
#     print('generate train data....')
#     model_train_data = generate_data(train_data, train_admin_features,train_item_features)
#     print('generate test data....')
#     model_test_data = generate_data(test_data, test_admin_features,test_item_features)
#     print('save file...')

    
    
    print('Save model train data...')
    with open('model_data/train_data.pkl', 'wb') as output_file:
        pickle.dump(model_train_data, output_file)
    print('Save model test data...')
    with open('model_data/test_data.pkl', 'wb') as output_file:
        pickle.dump(model_test_data, output_file)
    print('done....')
    
        
    