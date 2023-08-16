# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:33:04 2022

@author: m.dorosti
"""
import joblib
import traceback
import pandas as pd
import numpy as np
import sys
import json
from flask import Flask, request, render_template, session, redirect
from sklearn.cluster import KMeans
import pyodbc
import pickle
from collections import Counter


def preprocess_data(g):
               import pandas as pd
               import datetime as dt
               
               '''k_means = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
               n_clusters=4, n_init=10,  precompute_distances='auto',
               random_state=0, tol=0.0001, verbose=0)'''
              
               
              
               data=pd.DataFrame(g)     
               data.head()
               # Z score
               from scipy import stats
               z_score = np.abs(stats.zscore(data['SumPrice']))
                #threshold = 10
  
                # Position of the outlier
               a=np.where(z_score > 1000)
               #print("Old Shape: ", data.shape) 
               #data.drop(np.where(z > 4), inplace = True)
               data.drop(a[0], inplace = True)
               #print("New Shape: ", data.shape)
               #PRESENT = dt.datetime(2021,6,20)
               PRESENT = dt.datetime.today()
               #data[['Dt_Effect']]=int(data[['Dt_Effect']])
               data['Dt_Effect'] = pd.to_datetime(data['Dt_Effect'])
               

               #rfm= data.groupby('IdPrsClient').agg({'Dt_Effect': lambda date: (PRESENT - date.max()).days,
                                        #'CountIvc': lambda num: len(num),
                                        #'SumPrice': lambda price: price.sum()})
               rfm= data.groupby('IdPrsClient').agg({'Dt_Effect': lambda date: (PRESENT - date.max()).days,
                                        'CountIvc': lambda num: num.sum(),
                                        'SumPrice': lambda price: price.sum()})
  
  
               rfm.columns=['recency','frequency','monetary']
  
               rfm['recency'] = rfm['recency'].astype(int)


               rfm['r_quartile'] = pd.qcut(rfm['recency'],4, ['1','2','3','4'], duplicates='drop')
               rfm['r_charak']= pd.qcut(rfm['recency'],4 ,duplicates='drop')
               r_charak=rfm['r_charak'].unique()
               s=rfm.iloc[:,1]
               s = pd.qcut(s,4, duplicates='drop')
               #rfm['f_quartile'] = pd.qcut(rfm['frequency'],4, ['4','3','2','1'], duplicates='drop')
               rfm['f_quartile'] = pd.qcut(rfm['frequency'],4, ['4','3','2'], duplicates='drop')
               rfm['f_charak']= pd.qcut(rfm['frequency'],4 ,duplicates='drop')
               f_charak=rfm['f_charak'].unique()
               rfm['m_quartile'] = pd.qcut(rfm['monetary'],4, ['4','3','2','1'])
               rfm['m_charak']= pd.qcut(rfm['monetary'],4 ,duplicates='drop')
               m_charak=rfm['m_charak'].unique()

               #rfm['RFM_Scor♣e'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
               rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str)+ rfm.m_quartile.astype(str)
               
               
               #rfm['RFM_Score'] = rfm.r_quartile.astype(str)+rfm.m_quartile.astype(str)
               #rfm['RFM_Score'] =  rfm.m_quartile.astype(str)
               #p=rfm.head()
               #p=rfm[rfm['RFM_Score']=='121'].sort_values('monetary',ascending=False).head(5)
               p=rfm.sort_values('monetary',ascending=False)
               #p=rfm[rfm['RFM_Score']]
               #p=[p]
               df=rfm[['RFM_Score']]
                
               quotient_2d = np.array(df).reshape(-1,1)
            
               return quotient_2d,df,p,r_charak
           
def kmeansApp():
                 
        
            
            
               #myjson = request.json
               #myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               
               #json_=myjson['Array']
               '''ConnectionString=myjson['ConnectionString']
               #json_=request.json
               CustomerName=myjson['CustomerName']
               #CustomerName="PAYA2"
               s=ConnectionString.split(';')
               a=s[0].split('=')[1]
               b=s[1].split('=')[1]
               c=s[2].split('=')[1]
               d=s[3].split('=')[1]
               m='Sql Server'
               conn=pyodbc.connect(Driver=m,
                                   Server=a,
                                   Database=b,
                                   UID=c,
                                   PWD=d
                                   )'''
               
               
               conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=Modern_Master;'
                                  'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )
               
               
               with open('D://Dorosti//100.20python Service//GLOBAL PYTHON SERVICE//KmeansMosels//Sales Persons By Date.txt' ,'r')as file:
                   query=file.read()
                   
                   
               cursor = conn.cursor()   

               import pandas as pd
               GF=pd.read_sql_query(query,conn) 
                  
               
               json_=GF
               CustomerName="maryam" 
               import joblib
               import traceback
               import pandas as pd
               import datetime
               import numpy as np
               import datetime as dt
              
               import sys
               quotient_2d,df,p,r_charak=preprocess_data(json_)
               
               k_means=KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=400, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
               k_means.fit(quotient_2d)
               b=k_means.cluster_centers_
               k=[]
               for i in range(0,len(b)):
                   k.append(b[i][0])
                   
               l2=np.array(k)
               sort_index=np.argsort(l2)
               sort_index=list(sort_index)
               
                   
               counter=Counter(k_means.labels_)
               
               
               

               z = k_means.predict(quotient_2d)
               cluster_map = pd.DataFrame()
               cluster_map['data_index'] = df.index.values.astype(float)
               cluster_map['cluster'] =z.astype(float)
               Z=["1","2","3","4"]
               #f=[]
               
               #for i,j  in zip(sort_index,Z):
                  #f.append((i,j))
                  #if cluster_map['cluster'].any()==i:
                  #if cluster_map.iloc[]
                      #cluster_map['customer grade'].iloc[s]=j
                      
                   
                  #cluster_map['customer grade']=cluster_map['cluster'].replace({i:j+1,b:2,c:3,d:4})
                  #n.append(cluster_map['cluster'].replace({i:str(j)},inplace=False))
               cluster_map['customer grade']=cluster_map['cluster'].replace({sort_index[0]:Z[0],sort_index[1]:Z[1],sort_index[2]:Z[2],sort_index[3]:Z[3]},inplace=False)
               
               k1=[]
               k2=[]
               
               for i in range(0,len(cluster_map)):
                   k1.append(cluster_map.iloc[i,0])
                   k2.append(cluster_map.iloc[i,2])
                   
               cluster_map.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/KmeansMosels/'+str(CustomerName)+'CusromerClustering_model.csv')
               p.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/KmeansMosels/'+str(CustomerName)+'p.csv')   
                   
               filename='D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/KmeansMosels/'+str(CustomerName)+'CusromerClustering_model.sav'
               pickle.dump(k_means, open(filename, 'wb'))     
                   
               #return cluster_map.to_json(orient="records")
               conn = pyodbc.connect('Driver={SQL Server};Server=192.168.100.17\sql2019;Database=Modern_Master;uid=sa;pwd=PAYA+master')
               cursor = conn.cursor()
               cursor.execute("DELETE FROM bi.PrsClusters ;")
               conn.commit()
                  
               for i in range(0,len(cluster_map)):
           
             
             
                       cursor.execute("INSERT INTO bi.PrsClusters (IdPrs,ClusterIndex)  VALUES (?,?);"  ,k1[i],k2[i])
                       conn.commit()
               #return (str(r_charak[1]))
               return cluster_map

        
      
            
cluster_map=kmeansApp()  
print(cluster_map.head())
        
                     
            
           
            
           
            
           
            
           
            
           
            
           
            
           
