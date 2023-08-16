# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:16:49 2022

@author: m.dorosti
"""

import pandas as pd
import numpy as np
import joblib
import pandas as pd
import pyodbc
import dash_auth
import matplotlib.pyplot as plt
#import forecastingDash
import pyodbc 
from datetime import timedelta
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output 
with open('PathConfig.txt','r',encoding="utf-8") as file:
    
                   path=file.read()
           


dash.register_page(__name__, path='/forecasting')

#sf=pd.read_csv('PredictedSalesFromSqlJob.csv')
#sf=pd.read_csv('C:/Users/paya8/Desktop/jobs/PredictedSalesFromSqlJob.csv')
sf=pd.read_csv(path+'/jobs/PredictedSalesFromSqlJob.csv')

fig = go.Figure(data=[ go.Scatter(x=sf.date[ :len(sf)-20],y=sf.value[ :len(sf)-20], line=dict(color=' black', width=1), text='مقدار فروش',name='مقدار فروش' ),
                       go.Scatter(x=sf.date[len(sf)-20:len(sf)],y=sf.value[len(sf)-20:len(sf)], line=dict(color='red', width=1),text='مقدار پیش بینی شده فروش',name='مقدار پیش بینی شده فروش'),
                       go.Scatter(x=sf.date, y=sf.MA5, line=dict(color='orange', width=2), text='میانگین پنج روزه مقدار فروش',name='میانگین پنج روزه مقدار فروش'),
                       go.Scatter(x=sf.date, y=sf.MA20, line=dict(color='green', width=3),text='میانگین بیست روزه مقدار فروش',name='میانگین بیست روزه مقدار فروش')])


#sf.to_csv('C:/Users/paya8/Desktop/jobs/PredictedSalesFromGhadim.csv')
sf.to_csv(path+'/jobs/PredictedSalesFromGhadim.csv')
fig.update_layout(title_x=0.5,title_font_size=20)


# Plotting all increasing curves

#fig.show()
def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list

#-----------------------------END OF PLOT VALUES---------------------------------------------
INITIAL=50

layout=html.Div(children=[
    
    html.Div(children=[
        
        html.H1('*گزارش پیش بینی فروش*')
        #html.H2('پایا سافت')
        
        
        
        ], style={'textalign':'center'}),
    
    html.Div(children=[
        
     
        dcc.Graph(
        id='fig1',
        figure=fig)
        #html.Button("Train-Model!", id='Train-Model')
    #,
       
        
    #html.Div(
    #dcc.Graph(
        #id='fig2',
        #figure=fig2
    #)
     # )  
        
        
        
        ]#,style={'display':'flex'}
        
        )#,
    #html.Div(children=[
        
         #dcc.Graph(
        #id='fig3',
        #figure=fig3
   # )
    #,
    #dcc.Graph(
        #id='fig4',
       # figure=fig4
   # )
   
        
        
        
       # ], style={'display':'flex'})
      ##00BCD4  
    ],style = {'padding':'20px','backgroundColor' : 'black','textAlign':'center'})                     
       
    

def fetch_data_from_user_input(input_value):
    # code to query an API
    
   return ("heloo")






