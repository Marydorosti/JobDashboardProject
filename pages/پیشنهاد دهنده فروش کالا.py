# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:39:30 2022

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
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output 
from dash import Dash, dash_table
from dash import Dash, dcc, html, Input, Output, callback
PAGE_SIZE=5


#app = Dash(__name__)
dash.register_page(__name__, path='/RecommenderSystem')



CustomerItemdf=pd.read_excel('ExportCustomer-Item-CustItemRecommend.xlsx')
ItemItemdf=pd.read_excel('ExportItem-Item-data_neighbours.xlsx')


layout=html.Div(children=[
    
    html.Div(children=[
        
        html.H1('*پیشنهاد فروش کالا به مشتریان*')
        #html.H2('پایا سافت')
        
        
        
        ], style={'textalign':'center'}),
    
   
    html.Div(children=[
        
        dash_table.DataTable(
        id='datatable-CustomerItem',
        data=CustomerItemdf.to_dict('records'),
        columns=[{"name": i, "id": i} for i in CustomerItemdf.columns],
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
                'column_id': 'support',
            },
            #'backgroundColor': 'dodgerblue',
            'color': 'white'
        }
])
   
        
        
        
        ], style={'textalign':'center'})
        
        
        
    
    
    ],style = {'padding':'20px','backgroundColor' : '#00BCD4','textAlign':'center'})                     
       











                   
             
@callback(
    Output('datatable-CustomerItem', 'data'),
    Input('datatable-CustomerItem', "page_current"),
    Input('datatable-CustomerItem', "page_size"))
def update_table(page_current,page_size):
    return CustomerItemdf.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')


'''@callback(
    Output('datatable-ItemItem', 'data'),
    Input('datatable-ItemItem', "page_current"),
    Input('datatable-ItemItem', "page_size"))
def update_table(page_current,page_size):
    return ItemItemdf.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')'''














