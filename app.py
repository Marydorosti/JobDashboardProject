# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 08:09:24 2022

@author: m.dorosti
"""

from dash import Dash, html, dcc
import dash
import dash_auth
from dash import Dash, dcc, html, Input, Output, callback
import dash
import dash_labs as dl
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import flask
import os
from flask_login import login_user, LoginManager, UserMixin, logout_user, current_user
#import dash-auth and users dictionary
#import dash_auth
from users import USERNAME_PASSWORD_PAIRS
import os
from flask import Flask
from flask_login import login_user, LoginManager, UserMixin, current_user
import dash
from dash import dcc, html, Input, Output, State

#names=['a','b','c','d','e','f']

app = dash.Dash(
    __name__,  use_pages=True, suppress_callback_exceptions=True
)


auth = dash_auth.BasicAuth(
    app,
    USERNAME_PASSWORD_PAIRS
)

app.layout = html.Div([
	html.H1('داشبورد هوش تجاری پایا سافت'),
    

    html.Div(
        [
            html.Div(
               #html.Button("Go!",href=page["relative_path"]),
               
                dcc.Link(
                   #f"{page['name']} - {page['path']}", href=page["relative_path"]
                   html.Button(page['name']), href=page["relative_path"],style={'display': 'inline-block', 'vertical-align': 'middle',
                   #html.Button(names[0]), href=page["relative_path"],style={'display': 'inline-block', 'vertical-align': 'middle',
                  'min-width': '150px',
                   'height': '25px',
                   'margin-top': '0px',
                   'margin-left': '5px',
                   'color':'red'}
                )
            )
            for page in dash.page_registry.values()
        ],style = {'padding':'20px','backgroundColor' : '#00BCD4'}
    ),

	dash.page_container
],style = {'padding':'20px','backgroundColor' : '#00BCD4','textAlign':'center','font-family': 'Lucida Console','vertical-align': 'right','margin': {
                                        'l': 150, 'b': 20, 't': 0, 'r': 0
                                    }})













if __name__ == '__main__':


	app.run_server(debug=True)


#app.run_server(host='127.0.0.1', port='7080', proxy=None, debug=False, dev_tools_ui=None, dev_tools_props_check=None, dev_tools_serve_dev_bundles=None, dev_tools_hot_reload=None, dev_tools_hot_reload_interval=None, dev_tools_hot_reload_watch_interval=None, dev_tools_hot_reload_max_retry=None, dev_tools_silence_routes_logging=None, dev_tools_prune_errors=None, **flask_run_options)

