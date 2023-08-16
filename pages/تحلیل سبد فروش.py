# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:16:15 2022

@author: m.dorosti
"""

import pandas as pd
import numpy as np
import joblib
import traceback
import pandas as pd
from dash import Dash, dash_table
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from plotly.tools import mpl_to_plotly



df=pd.read_csv('AprioryJob.csv')
df=df[['گروه کالا1','گروه کالا2','lift']]

import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output 
PAGE_SIZE=5

#app = Dash(__name__)
dash.register_page(__name__, path='/apriory')
        
layout=html.Div(children=[
    
    html.Div(children=[
        
        html.H1('*تحلیل سبد فروش*')
        #html.H2('پایا سافت')
        
        
        
        ], style={'textalign':'center'}),
    
   
    html.Div(children=[
        
        dash_table.DataTable(
        id='datatable-apriory',
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
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
    Output('datatable-apriory', 'data'),
    Input('datatable-apriory', "page_current"),
    Input('datatable-apriory', "page_size"))
def update_table(page_current,page_size):
    return df.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')

        
        
        
        
        
        
        
        
 























    
   
       
