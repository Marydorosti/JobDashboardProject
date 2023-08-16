#In this Python file we create Forecasting Model for Sales,using time step 5 and predict for 20 days
#this is for Sql Server Job for plotly dashpredicted values
"""
Created on Sat Oct  1 12:37:55 2022

@author: m.dorosti
"""

# -*- coding: utf-8 -*-
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import joblib
from datetime import timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import os

with open('PathConfig.txt','r',encoding="utf-8") as file:
    
                   path=file.read()
           

#server_name='pc-dev-khodaie\sql2019'
#database_name='BI'
#database_name.table='dbo.GoodsPriceStock'
#cnxn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

#------------------------Insert Data From Sql Server------------------------------------------

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=192.168.10.199\sql2019;'
                      'Database=BI;'
                      'Trusted_Connection=no;'
                     'UID=sa ;'
                      'PWD=PAYA+master;')
sc = MinMaxScaler(feature_range = (0, 1))
cursor = conn.cursor()

#sql_query1 = pd.read_sql_query('SELECT * FROM dbo.DailySales',conn)
sql_query1 = pd.read_sql_query('SELECT * FROM dbo.DailySales WHERE IsPredict=0 ',conn)
#sql_query=pd.read_sql_query('DELETE FROM dbo.GoodsPriceStock WHERE IsPredict = 1')

sql_query=sql_query1[sql_query1.IsPredict != True]
sql_query.head()
sql_query=sql_query.iloc[:,3:9]
print(sql_query.head(5))
df = sql_query[['SalesTotalPrice']]
forecast_out=5

#-------------------------Detect Anomalies Function------------------------------------------
def anomaly_detect(df):
# Import IsolationForest
    from sklearn.ensemble import IsolationForest
# Assume that 13% of the entire data set are anomalies
 
    outliers_fraction = 0.01
    #outliers_fraction = 0.05
    #outliers_fraction = 0.001
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

#------------------------------------DropAnomalies Function---------------------------------------------------

def drop_anomalies(df,df4):
    import numpy as np
    ff=np.where(df4[0]==-1)
    ff=list(ff)
    for i in ff:
        df=df.drop(index=i)
    print(df)
    return df

df5=drop_anomalies(df5,df4)

#-------------------------------------CreatingDataSetWithTimeStep---------------------------------------

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
      
    
    
    return x_train, x_test, y_train, y_test,X__,Y__,y
    
#----------------Create Machine Learning Model For Regression-----------------------------------

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

    #print("model_test_accuracy:",model_test_accuracy)
    #print("model_train_accuracy:",model_train_accuracy)
    #print("model",sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)

    #pred=model.predict(x_test1[15:30])
    #predicted_price1 = sc.inverse_transform(pred.reshape(-1,1))
    #predicted_price1 = sc.inverse_transform(pred.reshape(-1,1))
                
      
    
    
    return model,model_train_accuracy,model_test_accuracy,x_test1,y_test,x_train1,y_train




 
#---------------Plot For Testing Model In Trainig part------------------------------------------------
'''def NewPrediction(x_test1,y_test,model,i):
    predicted_price=model.predict(x_test1[i:i+10])
    #predicted_price=model.predict(x_test1)
    plt.plot(y_test[i:i+10], color = 'red', label = 'real')
    #plt.plot(y_test, color = 'red', label = 'real')
    plt.plot(predicted_price, color = 'green', label = 'predicted')
    plt.title(' Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(' Stock Price')
    plt.legend()
    plt.show()'''
    
    
#------------------Calling create dataset for create new data set --------------------------------------------
x_train, x_test, y_train, y_test,X__,Y__,y=CreatingDataSetWithTimeStep(df5)

model,model_train_accuracy,model_test_accuracy,x_test1,y_test,x_train1,y_train=modelselection(x_train, x_test, y_train, y_test)

pred=model.predict(x_test1)
predicted_price1 = sc.inverse_transform(pred.reshape(-1,1))
#predicted_price1 = sc.inverse_transform(pred.reshape(-1,1))
model_error=mean_squared_error(y_test,predicted_price1)
#import math 
#print("model_error is :", model_error)
print("MSE =",mean_squared_error(y_test,pred))
print("RMSE =",np.sqrt(mean_squared_error(y_test,pred)))
print("MAE =",mean_absolute_error(y_test,pred))
print("RMAE =",np.sqrt(mean_absolute_error(y_test,pred)))
print("R2 =",r2_score(y_test,pred))


#print(x_train1.shape)
#print(x_train.shape)
#print(X__.shape)
#print(len(y))
#print(x_test.shape)

'''i=int(input("Enter i: "))
NewPrediction(x_test1,y_test,model,i=i)'''

#inputs=df5[len(df5)-25:len(df5)]
#print(inputs)
#inputs = np.array(inputs)
#inputs = np.array(inputs).reshape(-1,1)
#inputs = sc.transform(inputs)
'''X_test = []
for j in range(5, 25):
    X_test.append(inputs[j-5:j,:])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
#print(X_test.shape)
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price.reshape(-1,1))'''
#print(predicted_stock_price )
#print(len(predicted_stock_price))



#------------Save Model---------------------


joblib.dump(model,'PredictModel.pkl')


#------------Predict 20 Days After with giving 20 days as input-------------------

data=df5[['SalesTotalPrice']]
df5=sql_query[['PMonth','PSession','PDay','SalesAmount','InvoicesCount','SalesTotalPrice']] 
inputs=df5[len(df5)-25:len(df5)]
inputs = np.array(inputs)
#----------------------------------------------------------------------------------

X_test = []
for i in range(5, 25):
    X_test.append(inputs[i-5:i,:])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
#print(X_test.shape)
predicted_stock_price = model.predict(X_test)

#--------rescaling predicted values-------------------------------------------
scale=[df5[['SalesTotalPrice']].max(),df5[['SalesTotalPrice']].min()]
predicted_stock_price=predicted_stock_price *int(float(scale[0]))+(int(float(scale[1])))-int(float(scale[1]))


dataflag=[]
predictflag=[]
for i in range(0,len(data)):
    dataflag.append('past')
for i in range (0,len(predicted_stock_price)) :
    predictflag.append('forecasted')
flag=dataflag+predictflag 
  
dataframelist=df5['SalesTotalPrice'].to_list()+list(predicted_stock_price)
rollingave=df5['SalesTotalPrice'].rolling( 7).mean()

    
    
sql_query2 =pd.read_sql_query('SELECT MAX(Date) FROM dbo.DailySales  WHERE IsPredict=0',conn)
#Date=sql_query1[['Date']]
time=sql_query2.iloc[0]

timelist=[]
date=pd.to_datetime(time[0])
for i in range(0,len(predicted_stock_price)):
            #query2="DATEADD(DAY, i ,SELECT MAX(Dt_Effect) FROM dbo.GoodsPriceStock  WHERE IsPredict==0 as "
             date += timedelta(days=1)
             timelist.append(date)
#Date=time.to_list()+timelist
Date=sql_query1.iloc[:,0].to_list()+timelist
sf = pd.DataFrame([(a, b,c) for a, b,c in zip(Date,dataframelist,flag)],
                  columns=['date','value','flag'])

sf['MA5'] = sf.value.rolling(5).mean()
sf['MA20'] = sf.value.rolling(20).mean()

#sf.to_csv('C:/Users/paya8/Desktop/jobs/PredictedSalesFromSqlJob.csv')
sf.to_csv(path+'/PredictedSalesFromSqlJob.csv')


#sf.to_csv('PredictedSalesFromSqlJob.csv')
print(sf.tail())


'''if __name__ == '__main__':
    
   scheduler = BackgroundScheduler()
   scheduler.add_job(modelselection, 'interval',  minutes=2)
   scheduler.start()
   print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

   try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(2)
   except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()
'''








