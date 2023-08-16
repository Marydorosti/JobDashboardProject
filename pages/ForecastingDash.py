# -*- coding: utf-8 -*-


import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
from sklearn.metrics import mean_squared_error
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import joblib
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)


# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:32:55 2021

@author: m.dorosti
"""

import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
from sklearn.metrics import mean_squared_error

#server_name='pc-dev-khodaie\sql2019'
#database_name='BI'
#database_name.table='dbo.GoodsPriceStock'
#cnxn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)



########################InsertData&DropAnomalies###########################################################################




conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=192.168.10.199\sql2019;'
                      'Database=BI;'
                      'Trusted_Connection=no;'
                     'UID=sa ;'
                      'PWD=PAYA+master;')

cursor = conn.cursor()
#cursor.execute('SELECT * FROM dbo.GoodsPriceStock')
sql_query1 = pd.read_sql_query('SELECT * FROM dbo.DailySales',conn)
#sql_query=pd.read_sql_query('DELETE FROM dbo.GoodsPriceStock WHERE IsPredict = 1')
sql_query=sql_query1[sql_query1.IsPredict != True]
sql_query.head()
sql_query=sql_query.iloc[:,3:9]
print(sql_query.head(5))
df = sql_query[['SalesTotalPrice']]
#df=df.iloc[]
#df=sql_query
#df=sql
#print(sql_query)
#forecast_out =int(input("Enter forecast_out: "))
forecast_out=5
#print(type(sql_query))


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
    df['anomaly'] = pd.Series(df['anomaly'].values, index=df.index)
    a = df.loc[df['anomaly'] == -1] #anomaly
    
    return df4,df

df4,df=anomaly_detect(df)


df5=sql_query[['PMonth','PSession','PDay','SalesAmount','InvoicesCount','SalesTotalPrice']] 


def drop_anomalies(df,df4):
    import numpy as np

    #df=df[['Price']]

#df4=isolationforest.predict(df.values)
#df4=pd.DataFrame(df4)
    ff=np.where(df4[0]==-1)
    ff=list(ff)
    for i in ff:
        df=df.drop(index=i)
    print(df)
    return df

df5=drop_anomalies(df5,df4)


###############3333##############CreatingDataSetWithTimeStep##################33###########################################

def CreatingDataSetWithTimeStep(df5):
    
    y=df5['SalesTotalPrice']
    x=df5
    y=np.array(y)
    y = sc.fit_transform(y.reshape(-1,1))
    X__ = []
    Y__ = []
    for i in range(forecast_out,len(y)):
        X__.append(x[i-forecast_out:i])
        Y__.append(y[i])
    X__=np.array(X__)
    Y__=np.array(Y__)
    
    
    from sklearn.model_selection import train_test_split
    
    x_train, x_test, y_train, y_test = train_test_split(X__[0:len(X__)], Y__[0:len(Y__)], test_size=0.1,random_state=40) 
     # return  x_train, x_test, y_train, y_test,F_
    
#print(x_train.shape)    
    
    
    return x_train, x_test, y_train, y_test,X__,Y__,y
    















def modelselection(x_train, x_test, y_train, y_test):
    
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRFRegressor
    x_train1=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_test1=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
    #X_=X__.reshape(X__.shape[0],X__.shape[1]*X__.shape[2])
    model = XGBRFRegressor(n_estimators=150, random_state=42)
    model.fit(x_train1,y_train)
    #model.fit(X_,Y__)
    model_test_accuracy=model.score(x_test1,y_test)
    model_train_accuracy=model.score(x_train1,y_train)

    print("model_test_accuracy:",model_test_accuracy)
    print("model_train_accuracy:",model_train_accuracy)
    #print("model",sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)

    pred=model.predict(x_test1[15:30])
    predicted_price1 = sc.inverse_transform(pred.reshape(-1,1))
    predicted_price1 = sc.inverse_transform(pred.reshape(-1,1))
                
      
    
    
    return model,model_train_accuracy,model_test_accuracy,x_test1,y_test,x_train1,y_train

#model.predict5


#KNeighborsRegressor(...)    

def NewPrediction(x_test1,y_test,model,i):
    predicted_price=model.predict(x_test1[i:i+10])
    #predicted_price=model.predict(x_test1)
    plt.plot(y_test[i:i+10], color = 'red', label = 'real')
    #plt.plot(y_test, color = 'red', label = 'real')
    plt.plot(predicted_price, color = 'green', label = 'predicted')
    plt.title(' Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(' Stock Price')
    plt.legend()
    plt.show()
    
    

x_train, x_test, y_train, y_test,X__,Y__,y=CreatingDataSetWithTimeStep(df5)

model,model_train_accuracy,model_test_accuracy,x_test1,y_test,x_train1,y_train=modelselection(x_train, x_test, y_train, y_test)
pred=model.predict(x_test1)
predicted_price1 = sc.inverse_transform(pred.reshape(-1,1))
#predicted_price1 = sc.inverse_transform(pred.reshape(-1,1))
model_error=mean_squared_error(y_test,predicted_price1)
#import math 
print("model_error is :", model_error)
#model_error*88

print(x_train1.shape)
print(x_train.shape)
print(X__.shape)
print(len(y))






print(x_test.shape)

i=int(input("Enter i: "))
NewPrediction(x_test1,y_test,model,i=i)

inputs=df5[len(df5)-25:len(df5)]
print(inputs)
inputs = np.array(inputs)
#inputs = np.array(inputs).reshape(-1,1)
#inputs = sc.transform(inputs)
X_test = []
for i in range(5, 25):
    X_test.append(inputs[i-5:i,:])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
print(X_test.shape)
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price.reshape(-1,1))
print(len(predicted_stock_price))

joblib.dump(model,'PredictModel.pkl')




#for i in range(0,len(predicted_stock_price)):
    
   #print(predicted_stock_price[i])


#a=[2,5,6,7,0]
#b=[6,7,8,9,0]
#def sum(a,b):
    #return a+b


#predicted_stock_price=sum(a,b)



'''df=pd.DataFrame(predicted_stock_price)
   

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd


app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


fig = px.bar(df)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children=
        "Dash: A web application framework for your data."
    ),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True,port=8051)'''







#'''q2='''SELECT MAX(Dt_Effect) FROM dbo.GoodsPriceStock  WHERE IsPredict==0;''' %int()






