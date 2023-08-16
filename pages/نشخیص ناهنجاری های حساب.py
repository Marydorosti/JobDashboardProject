# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:30:52 2022

@author: m.dorosti
"""

import pyodbc
import pandas as pd
import dash

import plotly.express as px

import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output 

#server_name='pc-dev-khodaie\sql2019'
#database_name='BI'
#database_name.table='dbo.GoodsPriceStock'
#cnxn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
with open('PathConfig.txt','r',encoding="utf-8") as file:
    
                   path=file.read()
           


dash.register_page(__name__, path='/AnomalyDetection')




conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=192.168.10.199\sql2019;'
                      'Database=BI;'
                      'Trusted_Connection=no;'
                     'UID=sa ;'
                      'PWD=PAYA+master;')

cursor = conn.cursor()
#cursor.execute('SELECT * FROM dbo.GoodsPriceStock')
sql_query = pd.read_sql_query('SELECT * FROM dbo.DailySales',conn)
#sql_query=pd.read_sql_query('DELETE FROM dbo.GoodsPriceStock WHERE IsPredict = 1')
sql_query=sql_query[sql_query.IsPredict != True]
#print(sql_query.head())
sql_query=sql_query.iloc[:,0:9]
#print(sql_query.head(5))
df = sql_query[['SalesTotalPrice']]




forecast_out=20




def anomaly_detect(df):
# Import IsolationForest
    from sklearn.ensemble import IsolationForest
# Assume that 13% of the entire data set are anomalies
 
    outliers_fraction = 0.01
    isolationforest =  IsolationForest(contamination=outliers_fraction)
    
    isolationforest.fit(df.values)
    df4=isolationforest.predict(df.values)
    df4=pd.DataFrame(df4)
    df['anomaly'] = pd.Series(isolationforest.predict(df.values))
  # visualization
    #df['anomaly'] = pd.Series(df['anomaly'].values, index=df.index)
    a = df.loc[df['anomaly'] == -1] #anomaly
    
    #----------- visualization--------------------
    #df['anomaly'] = pd.Series(df['anomaly'].values, index=df.index)
    '''a = df.loc[df['anomaly'] == -1] #anomaly
    _ = plt.figure(figsize=(18,6))
    _ = plt.plot(df, color='blue', label='Normal')
    _ = plt.plot(a, linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
    _ = plt.xlabel('Date and Time')
    _ = plt.ylabel('Sensor Reading')
    _ = plt.title('dataset Anomalies')
    _ = plt.legend(loc='best')'''
    
   
    
    
    
    
    #fig=plt.gcf()
    #plt.show();
    return df4,df

df4,df=anomaly_detect(df)

sql_query['anomaly'] = df['anomaly']
#print(sql_query.head())

anomalyfig=px.scatter(sql_query,x='Date',y='SalesTotalPrice',color='anomaly')
#fig.show()

layout=html.Div(children=[
    
    html.Div(children=[
        
        html.H1('*تشخیص ناهنجاری ها*'),
        html.H2('پایا سافت')
        
        
        
        ], style={'textalign':'center'}),
    
    html.Div(children=[
        
     
        dcc.Graph(
        id='anomalyfig',
        figure=anomalyfig)
        #html.Button("Train-Model!", id='Train-Model')
        ], style={'display':'center'})
        
    ],style = {'padding':'20px','backgroundColor' : '#00BCD4','textAlign':'center'})                     
       







