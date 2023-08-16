# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:56:04 2022

@author: m.dorosti
"""
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt
from dash import Dash, dash_table
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from plotly.tools import mpl_to_plotly
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
#%matplotlib inline
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pylab import rcParams
rcParams['figure.figsize'] = 9, 8  # set plot size


plot_data_frame=pd.read_csv('Clusteringplot_data_frame.csv')
cluster_map = pd.read_csv('CustomerClusteringFromSqlJob.csv') 


fig3 = px.scatter(plot_data_frame,x='column1',y='column2',color='labels',title='خوشه های مشتریان') 
                           
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output 


#app = Dash(__name__)
dash.register_page(__name__, path='/kmeans')

fig=px.pie(cluster_map,names='گرید مشتری',values='خوشه',title='نمودار دایره ای خوشه بندی مشتریان')
fig.show()

PAGE_SIZE = 5

layout=html.Div(children=[
    
    html.Div(children=[
        
        html.H1('*خوشه بندی مشتریان*'),
        #html.H2('پایا سافت')
        
        
        
        ], style={'textalign':'center'}),
    
    html.Div(children=[
        
     
        dcc.Graph(
        id='fig1',style={
            'height': 500,
           'weight':500
        },
        figure=fig)
        #html.Button("Train-Model!", id='Train-Model')
    ,
       
        
    #html.Div(
    dcc.Graph(
        id='fig3'
        ,style={
            'height': 500,
           'weight':500
        },
        
        
        figure=fig3
    )
      #) 
        
        
        
        ],style={ 'display': 'center','position': 'relative'}
        
        ),
    html.Div(children=[
        
        dash_table.DataTable(
        id='datatable-paging',
        data=cluster_map.to_dict('records'),
        columns=[{"name": i, "id": i} for i in cluster_map.columns],
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
    Output('datatable-paging', 'data'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"))
def update_table(page_current,page_size):
    return cluster_map.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')
















