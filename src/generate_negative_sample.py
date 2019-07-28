# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import randint, sample


if __name__=='__main__':
    print('read csv files')
    train_data = pd.read_csv('../csv/Antai_AE_round1_train_20190626.csv')
    item_data = pd.read_csv('../csv/Antai_AE_round1_item_attr_20190626.csv')

    print('挑选yy国家或是xx国家')
    train_data = train_data[train_data['buyer_country_id'].isin(['yy','xx'])]
    dfyy = train_data.copy()
    del dfyy['buyer_country_id']
    del dfyy['create_order_time']
    del dfyy['irank']
    dffyy = dfyy.groupby(['buyer_admin_id','item_id']).agg('sum').reset_index()

    print('将用户购买的商品和店对应起来')
    item_store = item_data.copy()
    del item_store['cate_id']
    del item_store['item_price']
    item_store['itemNumber'] = 1
    user_store = pd.merge(item_store,dffyy,how = 'left',on = 'item_id')
    user_store =  user_store.dropna(axis = 0,how = 'any')

    print("统计商店和商店的总的商品id")
    itemInStore = pd.DataFrame()
    a = list(set(user_store['store_id'].to_list()))
    itemInStore['store_id'] = a
    a = itemInStore['store_id'].to_list()
    b = []
    for i in tqdm(range(len(a))):
        store = item_data[item_data['store_id'].isin([a[i]])]
        b.append(store['item_id'].to_list())
    itemInStore['storeItems'] = b
    del user_store['item_id']
    user_store = user_store.groupby(['store_id','buyer_admin_id']).agg('sum').reset_index() #统计用户在商店里买了多少东西

    userStoreItems = pd.merge(user_store,itemInStore,how = 'left',on = 'store_id') #将商店里的商品加到用户和商店后面

    print('按照用户在一个店里的购买的数量，不放回的随机取相同数量样本作为负样本')
    a = userStoreItems['itemNumber'].to_list()
    b = userStoreItems['storeItems'].to_list()
    c = []
    for i in tqdm(range(len(a))):
        c.append(sample(b[i],a[i]))
    userStoreItems['faultSample'] = c

    print('取出来的是一个list，采用写入的方式快速的使用户和对应取出的样本一一对应')
    file_comment = './FaultSample.txt'
    with open(file_comment,'w') as fileobject:
        fileobject.write(str(1) + ',' + str(2) + '\n')
        for i in tqdm(range(len(userStoreItems))):
            result = userStoreItems.iloc[i,4]
            for r in result:
                fileobject.write(str(r) + ',' + str(userStoreItems.iloc[i,1]) + '\n')
    fileobject.close()

    print('将写入的txt文件打开，重命名')
    FaultSample = pd.read_csv('./FaultSample.txt')
    FaultSample.columns = ['itemid','buyer_admin_id']
    FaultSample['label'] = 0 #负样本标签
    FaultSample.to_csv('negative_sample.csv',index=False)
    print('done')
    # #下面取正样本
    # TrueSample = pd.DataFrame()
    # TrueSample['itemid'] = dffyy['item_id']
    # TrueSample['buyer_admin_id'] = dffyy['buyer_admin_id']
    # TrueSample['label'] = 1

    # #合并正负样本，并去重
    # sample = pd.concat([TrueSample,FaultSample],axis = 0)
    # sample = sample.drop_duplicates(subset = ['itemid','buyer_admin_id'],keep = 'first')
    # sample['buyer_id'] = sample['buyer_admin_id'].apply(lambda x: int(x)) #将浮点型数据改为整型
    # del sample['buyer_admin_id']
    # sample.to_csv('sample.csv',index = False)