# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:39:49 2022

@author: m.dorosti
"""



#import modules
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt
#import plotly.express as px  # (version 4.7.0 or higher)
#import plotly.graph_objects as go
with open('PathConfig.txt','r',encoding="utf-8") as file:
    
                   path=file.read()
           
def datapreprocess(excel_path):
  data = pd.read_excel(excel_path, converters={'NodeId':str})
  data.head()


  # Z score
  from scipy import stats
  import numpy as np
  
  z = np.abs(stats.zscore(data['SumPrice']))


  #threshold = 10
  
# Position of the outlier
  a=np.where(z > 10)


    
  print("Old Shape: ", data.shape) 
#data.drop(np.where(z > 4), inplace = True)
  data.drop(a[0], inplace = True)
  print("New Shape: ", data.shape)
  PRESENT = dt.datetime(2021,6,20)

  rfm= data.groupby('IdPrsClient').agg({'Dt_Effect': lambda date: (PRESENT - date.max()).days,
                                        'CountIvc': lambda num: len(num),
                                        'SumPrice': lambda price: price.sum()})
  
  rfm.columns=['recency','frequency','monetary']
  
  rfm['recency'] = rfm['recency'].astype(int)


  rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4'])
  rfm['f_quartile'] = pd.qcut(rfm['frequency'], 5, ['4','3','2','1'],duplicates='drop')
  rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])

  rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
  #rfm.head(50)
  df=rfm[['RFM_Score']]
  return(df,data)
df,data=datapreprocess(excel_path='AsreJadid Sales Persons By Date.xlsx')
print(df)


from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

#%matplotlib inline
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA # CA from PCA function
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pylab import rcParams
rcParams['figure.figsize'] = 9, 8  # set plot size

# k-means cluster analysis for 1-15 clusters                                              
from scipy.spatial.distance import cdist
clusters=range(1,15)
meandist=[]

# loop through each cluster and fit the model to the train set
# generate the predicted cluster assingment and append the mean 
# distance my taking the sum divided by the shape
'''for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(df)
    clusassign=model.predict(df)
    meandist.append(sum(np.min(cdist(df, model.cluster_centers_, 'euclidean'), axis=1))
    / df.shape[0])'''
    
k_means=KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')



quotient_2d = np.array(df).reshape(-1,1)
k_means.fit(quotient_2d)
z = k_means.predict(quotient_2d) 
cluster_map = pd.DataFrame() 
#cluster_map['data_index'] = df.index.values
#cluster_map['cluster'] =z
#cluster_map['customer grade']=cluster_map['cluster'].replace({0:"d",1:"a",2:"c",3:"b"})
cluster_map['نام مشتری'] = df.index.values
cluster_map['خوشه'] =z
cluster_map['گرید مشتری']=cluster_map['خوشه'].replace({0:"d",1:"a",2:"c",3:"b"})               
              
              

print(cluster_map)

pca_2 = PCA(2) # return 2 first canonical variables
plot_columns = pca_2.fit_transform(data[['CountIvc','CountGds','SumAmount','SumPrice']]) 

plot_data_frame = pd.DataFrame([(a, b,c) for a, b ,c in zip(plot_columns[:,0],plot_columns[:,1],k_means.labels_)],
                  columns=['column1','column2','labels'])

import joblib

joblib.dump(k_means, 'KmensModel.pkl')
print("Model dumped!") 
#k_means = joblib.load('KmeansModel.pkl') 
#joblib.dump(datapreprocess,'function.pkl') 
#datapreprocess=joblib.load('function.pkl')
#print("function dumped!")
#cluster_map.to_csv('CustomerClusteringFromSqlJob.csv')
#plot_data_frame.to_csv('Clusteringplot_data_frame.csv')
cluster_map.to_csv(path+'/CustomerClusteringFromSqlJob.csv')
plot_data_frame.to_csv(path+'/Clusteringplot_data_frame.csv')



'''cluster_map = pd.DataFrame()
cluster_map['data_index'] = df.index.values
cluster_map['cluster'] = k_means.labels_
#cluster_map[cluster_map.cluster == 3]

cluster_map['customer grade']=cluster_map['cluster'].replace({0:"d",1:"a",2:"c",3:"b"})
cluster_map.to_csv('customerclustering.csv')'''
#fig=px.pie(cluster_map,names='customer grade',values='cluster')
#fig.show()












