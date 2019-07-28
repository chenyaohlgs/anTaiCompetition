import numpy as np
import pickle
import pandas as pd

def readcsv_save_pickle():
    
    #读取item
    output = open(ITEMATTR_PKL_PATH, 'wb')
    item_attr=pd.read_csv(ITEMATTR_CSV_PATH)
    pickle.dump(item_attr, output)
    output.close()
    #读取train  
    output = open(TRAIN_PKL_PATH, 'wb')
    train=pd.read_csv(TRAIN_CSV_PATH)
    train['create_order_time'] = train.create_order_time.apply(lambda x:pd.to_datetime(x))
    train['hour']=train['create_order_time'].dt.hour
    train['date']=train['create_order_time'].dt.day
    train['month']=train['create_order_time'].dt.month
    train['year']=train['create_order_time'].dt.year
    train['month-date'] = train.month.astype(str)+'-'+train.date.astype(str)
    train['count'] = 1
    train['dayofweek']=train['create_order_time'].dt.dayofweek
    train['isweekend']=train['dayofweek'].apply(lambda x:0 if x<5 else 1)
    pickle.dump(train, output)
    output.close()
    #读取test
    output = open(TEST_PKL_PATH, 'wb')
    test=pd.read_csv(TEST_CSV_PATH)
    test['create_order_time'] = test.create_order_time.apply(lambda x:pd.to_datetime(x))
    test['hour']=test['create_order_time'].dt.hour
    test['date']=test['create_order_time'].dt.day
    test['month']=test['create_order_time'].dt.month
    test['year']=test['create_order_time'].dt.year
    test['month-date'] = test.month.astype(str)+'-'+test.date.astype(str)
    test['count'] = 1
    test['dayofweek']=test['create_order_time'].dt.dayofweek
    test['isweekend']=test['dayofweek'].apply(lambda x:0 if x<5 else 1)
    pickle.dump(test, output)
    output.close()

def drop_lose_item(data_df,item_df, lose_item):
    length = len(list(lose_item))
    all_drop_index= []
    for i,item in enumerate(list(lose_item)):
        drop_index = data_df[data_df['item_id'] == item].index[:]
        all_drop_index.extend(list(drop_index.values))
    data_df_filter = data_df.drop(all_drop_index).reset_index(drop=True)  # 去掉item不在商品表中的数据
    data_df_filter = pd.merge(data_df_filter,item_df, how='left',on='item_id') # 将商品表中的数据添加到原始dataframe中   
    return data_df_filter

def country_class(data_df):
    data_df['buyer_country_id'][data_df['buyer_country_id'] == 'xx'] =1
    data_df['buyer_country_id'][data_df['buyer_country_id'] == 'yy'] =2
    data_df['buyer_country_id'][data_df['buyer_country_id'] == 'zz'] =3
    return data_df

def buy_time_distribute(data_df):
    #计算用户购买商品的时间分布情况
    # 0-8: 1, 8-16:2. 16-24:3
    a = data_df['hour'].to_list()
    b = []
    for i in range(len(a)):
        if a[i]>=0 and a[i]<8:
            b.append(1)
        elif a[i]>=8 and a[i]<16:
            b.append(2)
        elif a[i]>=16 and a[i]<24:
            b.append(3)
    data_df['time_cluster'] = b
    time_cluster = pd.get_dummies(data_df['time_cluster'])
    data_df = pd.concat([data_df,time_cluster],axis=1)
    data_df = data_df.rename(columns={1:'shoptime_1',2:'shoptime_2',3:'shoptime_3'})
    return data_df

def record_popular_item(data_df):
    # 合并热销商品id到训练集中
    popular_item_df = pd.DataFrame(columns=['item_id','if_popular'])
    popular_item_df['item_id'] = popular_item_idx_train
    popular_item_df['if_popular'] = 1
    data_df = pd.merge(data_df,popular_item_df,how='left',on='item_id')
    data_df['if_popular'].fillna(0, inplace=True)
    return data_df


def generate_user_features(data_df):
    admin_df = pd.DataFrame(columns=['buyer_admin_id',  #用户id,
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
                           'shoptime_3',  #在16-24时间段...
                          ])
    buyer_id = np.sort(data_df['buyer_admin_id'].unique())
    print('add item id....')
    admin_df['buyer_admin_id'] = buyer_id  # id填充
    print('add num_shop....')
    admin_df['num_shop'] = data_df.groupby(['buyer_admin_id'])['count'].sum().to_list()  #，每个admin购买次数
    print('add num_item....')
    admin_df['num_item'] = data_df[['buyer_admin_id','item_id','count']].drop_duplicates().groupby(['buyer_admin_id'])['count'].sum().to_list() # 商品件数
    print('add pop_unpop_rate...')
    tmp = data_df.groupby(['buyer_admin_id'])['if_popular'].sum()/data_df.groupby(['buyer_admin_id'])['count'].sum()
    admin_df['pop_unpop_rate'] = tmp.to_list()
    #商品价格
    print('add item price....')
    admin_df['item_price_sum'] = data_df.groupby(['buyer_admin_id'])['item_price'].sum().to_list()
    admin_df['item_price_max'] = data_df.groupby(['buyer_admin_id'])['item_price'].max().to_list()
    admin_df['item_price_min'] = data_df.groupby(['buyer_admin_id'])['item_price'].min().to_list()
    admin_df['item_price_mean'] = data_df.groupby(['buyer_admin_id'])['item_price'].mean().to_list()
    admin_df['item_price_median'] = data_df.groupby(['buyer_admin_id'])['item_price'].median().to_list()
    print('add num_item_cate....')
    admin_df['num_item_cate'] = data_df[['buyer_admin_id','cate_id','count']].drop_duplicates().groupby(['buyer_admin_id'])['count'].sum().to_list()
    print('add num_store_cate....')
    admin_df['num_store_cate'] = data_df[['buyer_admin_id','store_id','count']].drop_duplicates().groupby(['buyer_admin_id'])['count'].sum().to_list()
    print('add shop time class....')
    # 各时间段购买商品数量
    admin_df['shoptime_1'] = data_df.groupby(['buyer_admin_id'])['shoptime_1'].sum().to_list()
    admin_df['shoptime_2'] = data_df.groupby(['buyer_admin_id'])['shoptime_2'].sum().to_list()
    admin_df['shoptime_3'] = data_df.groupby(['buyer_admin_id'])['shoptime_3'].sum().to_list()
    return admin_df

def generate_item_features(data_df,item_df):
    buyed_item_df = pd.DataFrame(columns=['item_id','num_sell'])
    buyed_item_df['item_id'] = data_df['item_id'].unique()
    buyed_item_df['num_sell'] = data_df.groupby(['item_id'])['count'].sum().to_list()
    buyed_item_df = pd.merge(buyed_item_df, item_df, on='item_id',how='left')
    return buyed_item_df
    
if __name__=='__main__':
    saved_pickle = True  #如果已经保存生成过pickle文件，则为True， 如果需要生成pickle文件，则设置为False
    
    ITEMATTR_CSV_PATH = '../csv/Antai_AE_round1_item_attr_20190626.csv'
    TRAIN_CSV_PATH = '../csv/Antai_AE_round1_train_20190626.csv'
    TEST_CSV_PATH = '../csv/Antai_AE_round1_test_20190626.csv'
    ITEMATTR_PKL_PATH = '../data/item_attr.pkl'
    TRAIN_PKL_PATH = '../data/train.pkl'
    TEST_PKL_PATH = '../data/test.pkl'
    if not saved_pickle:
        print('read csv file and generate pickle files...')
        readcsv_save_pickle()
    print('read pickle files directly...')
    item_df =  pickle.load(open(ITEMATTR_PKL_PATH, 'rb'), encoding='iso-8859-1')
    train_df =  pickle.load(open(TRAIN_PKL_PATH, 'rb'), encoding='iso-8859-1')
    test_df =  pickle.load(open(TEST_PKL_PATH, 'rb'), encoding='iso-8859-1')
    print('calculate popular item...')
    popular_item_intrain = train_df.groupby(['item_id'])['count'].sum().reset_index().sort_values('count',ascending=False)
    popular_item_idx_train = popular_item_intrain.iloc[:,0][popular_item_intrain.iloc[:,1].cumsum() <= int(train_df.shape[0] * 0.15)].values
    print('热销商品的数量为：%s'%(popular_item_idx_train.shape[0]))
    print('处理缺失值.....')
    train_lose_item = set(train_df['item_id'].unique()).difference(set(item_df['item_id'].unique()))
    test_lose_item = set(test_df['item_id'].unique()).difference(set(item_df['item_id'].unique()))
    print('训练集中商品数 %s, 商品表中商品数 %s'%(train_df['item_id'].unique().shape[0], item_df['item_id'].unique().shape[0]))
    print('训练集中不在商品表中的商品数：%s'%(len(train_lose_item)))
    print('测试集中不在商品表中的商品数：%s'%(len(test_lose_item)))
    print('drop lose item .....')
    train_filter = drop_lose_item(train_df, item_df, train_lose_item)
    test_filter = drop_lose_item(test_df, item_df, test_lose_item)
    print('calculate time distribution...')
    train_filter = buy_time_distribute(train_filter)  # 购买时间分布情况
    test_filter = buy_time_distribute(test_filter)
    print('record popular item...')
    train_filter = record_popular_item(train_filter)
    test_filter = record_popular_item(test_filter)
    print('country....')
    train_filter = country_class(train_filter)
    test_filter = country_class(test_filter)
    print('生成训练集用户画像...')
    train_admin_df = generate_user_features(train_filter)
    print('生成测试集用户画像...')
    test_admin_df = generate_user_features(test_filter)
    print('生成训练集中被购买商品特征...')
    train_item_features_df = generate_item_features(train_filter, item_df)
    print('生成测试集中被购买商品特征...')
    test_item_features_df = generate_item_features(test_filter, item_df)
    print('Save pickle file in /features ...')
    print('...')
    print('Save train admin features...')
    with open('features/train_admin_features.pkl', 'wb') as output_file:
        pickle.dump(train_admin_df, output_file)
    print('Save test admin features...')
    with open('features/test_admin_features.pkl', 'wb') as output_file:
        pickle.dump(test_admin_df, output_file)
    print('Save train item features...')
    with open('features/train_item_features.pkl', 'wb') as output_file:
        pickle.dump(train_item_features_df, output_file)
    print('Save test item features...')
    with open('features/test_item_features.pkl', 'wb') as output_file:
        pickle.dump(test_item_features_df, output_file)
    print('..Done!!')

    
    
          
        
    
    

    
    
    
    
    
    
    
    
    
    