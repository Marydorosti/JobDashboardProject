# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 08:43:19 2021

@author: m.dorosti
"""

# Dependencies
#from flask import Flask, request, jsonify
#import joblib
#import traceback
import pandas as pd
import numpy as np
##from flask import Flask, request, render_template, session, redirect
#from sklearn.impute import SimpleImputer
#import skl2onnx
#import onnx
#import sklearn
#from sklearn.linear_model import LogisticRegression
import numpy
import onnxruntime as rt
#from skl2onnx.common.data_types import FloatTensorType
#from skl2onnx import convert_sklearn
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#import sys
#import pyodbc
from dash import Dash, html, dcc
import dash
#import dash_auth
from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px

with open('PathConfig.txt','r',encoding="utf-8") as file:
    
                   path=file.read()
           
    

def preprocess_query(json_,CustomerName):
    #my_file = open(str(CustomerName)+"2removedFeatures.txt.txt", "r")
    #my_file = open(str('C:/Users/paya8/Desktop/Dash-by-Plotly-master/my multi page dash/pages/'+CustomerName)+"1bestFeatures.txt", "r")
    my_file = open(str(path+'/pages/'+CustomerName)+"1bestFeatures.txt", "r")
    #to_drop = my_file.readline()
    to_drop=my_file.read().splitlines()
    sf=pd.DataFrame(json_,columns=['column1','column2','column3'])
   
    
    #to_drop=list(to_drop)
    s=pd.DataFrame(json_)
    s=s.iloc[:,3:]
    p=type(s)
     
    #query1=s.drop(s[to_drop], axis=1)
    s=s[to_drop]
    #s.columns=to_drop
    #query3=s.drop(columns=['PrsType','IsDishonor'])
    #f=query3.columns
    #query3=s
    m=len(s.columns)
    #query3=query3.iloc[:,0:m]
    r=s.iloc[:,0:m].columns
    f=s.columns
    s=s.iloc[:,0:m].values
    
    
    return to_drop,s,f,m,p,to_drop,r
    
def Predict1CheckApi():
    
            
            
            
            '''conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=192.168.10.199\sql2019;'
                      'Database=BI;'
                      'Trusted_Connection=no;'
                     'UID=sa ;'
                      'PWD=PAYA+master;')

            cursor = conn.cursor()'''
            json_=pd.read_excel('DishonorChqs.xlsx')
            json_=json_[0:15]
            #myjson = request.json
            #myjson2=pd.DataFrame(request.json)
              
               
            #json_=myjson['Array']
            #CustomerName=myjson['CustomerName']
            CustomerName="Aramesh"
            
            to_drop,s,f,m,p,to_drop,r=preprocess_query(json_,CustomerName)
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler(feature_range = (0, 1))
            sc = sc.fit(s)
            
            s=sc.transform(s)
            sess = rt.InferenceSession(path+'/pages/'+str(CustomerName)+'1model.onnx')
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            pred_onx = sess.run([label_name],
                    {input_name: s.astype(numpy.float32)})[0]
           
            #return jsonify({'prediction': str(pred_onx )})
            return pred_onx,json_
            #return  '{} {} '.format(pred_onx,p) 
            
            
pred_onx,json_=Predict1CheckApi()
#json_=json_[:,3]
dataframe = pd.DataFrame([(a, b,c) for a, b,c in zip(json_['Amount'].to_list(),json_['DueDays'].to_list(),pred_onx)],
                  columns=['مبلغ چک','تعداد روز باقی مانده','وضعیت پیش بینی'])

#json_
#pred_onx=pd.DataFrame(pred_onx,columns=['column1'])
fig = px.scatter(dataframe,x='مبلغ چک',y='تعداد روز باقی مانده',color='وضعیت پیش بینی',title='cheque Classification')
#fig=px.scatter(json_,x='json_',y='json_',color='pred-onx')

import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
from dash import Dash, dash_table
from dash import Dash, dcc, html, Input, Output
import numpy as np
import matplotlib.pyplot as plt 

      
            #return jsonify({'trace': traceback.format_exc()})
dash.register_page(__name__, path='/cheque')



PAGE_SIZE = 5

layout=html.Div(children=[
    
    html.Div(children=[
        
        html.H1('*وضعیت چک های حساب*'),
        html.H2('پایا سافت')
        
        
        
        ], style={'textalign':'center'}),
    
    html.Div(children=[
        
     
        
        dcc.Graph(
        id='chequefig',
        figure=fig)
        
    
        
        
        
        ],style={'display':'center'}
        
        ),
    html.Div(children=[
        
        dash_table.DataTable(
        id='datatable-cheque',
        data=dataframe.to_dict('records'),
        columns=[{"name": i, "id": i} for i in dataframe.columns],
        page_current=0,
        page_size=PAGE_SIZE,
        page_action='custom',
        style_cell={'textAlign': 'center'},
        style_data={ 'border': '2px solid blue' },
        style_header={ 'border': '2px solid pink' },
        editable=True,
    style_data_conditional=[
        {
            'if': {
                'column_id': 'customer grade',
            },
            #'backgroundColor': 'dodgerblue',
            'color': 'white'
        }
])
   
        
        
        
        ], style={'textalign':'center'})
        
        
        
    
    
    ],style = {'padding':'20px','backgroundColor' : '#00BCD4','textAlign':'center'}) 

@callback(
    Output('datatable-cheque', 'data'),
    Input('datatable-cheque', "page_current"),
    Input('datatable-cheque', "page_size"))
def update_table(page_current,page_size):
    return dataframe.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')













                    
           

