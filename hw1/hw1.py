# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:34:36 2021

@author: 劉之岳
"""

import numpy as np
import pandas as pd
import time
df = pd.read_csv('homework.csv')
# -------------- Numpy --------------
arr = np.array(df, dtype = 'float')

start = time.time()

""" Your Code Here: 用 Numpy 計算 “homework.csv” 中每個 feature 的平均值、中位數、最大值、最小值，並比較兩者運算時間"""
mean=np.mean(arr,axis=0) #col mean
median=np.median(arr,axis=0)
maxium=np.max(arr,axis=0)
minium=np.min(arr,axis=0)
end = time.time()
print(f'Numpy has done in {(end - start):.4f} sec.')

# -------------- Pandas --------------
start = time.time()
""" Your Code Here: 用 Pandas 計算 “homework.csv” 中每個 feature 的平均值、中位數、最大值、最小值，並比較兩者運算時間"""
mean=df.mean()
median=df.median()
maxium=df.max()
minium=df.min()
end = time.time()
print(f'Pandas has done in {(end - start):.4f} sec.')

"""計算各個 sample 與 Vec 之距離"""
Vec=mean
dist = [] 
for i in range(0,len(df)):
    dist.append(np.linalg.norm(Vec-df.iloc[i]))#第i+1筆資料 歐拉距離

"""計算 distance 的 mean, std, 並篩選離群值"""
dist_mean=np.mean(dist)
dist_std=np.std(dist)

""" 刪除離群值"""
drop_num=[] #收集需要被刪除的房子序號
for i, element in enumerate(dist):
    if np.absolute(element-dist_mean)>(3*dist_std):
        drop_num.append(i)
df_after_delete=df.copy()
df_after_delete.drop(drop_num,axis=0) #刪除離群值房子序號