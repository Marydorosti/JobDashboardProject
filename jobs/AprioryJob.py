import pandas as pd
#from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
#import numpy as np
#from flask import Flask, request, render_template, session, redirect
#import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt
#from dash import Dash, dash_table
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
#from dash import Dash, dcc, html, Input, Output, callback
from plotly.tools import mpl_to_plotly

with open('PathConfig.txt','r',encoding="utf-8") as file:
    
                   path=file.read()
           



def preprocess_dataset():
    #dataset=pd.DataFrame(json_)
    dataset=pd.read_excel('Goods In Invoices.xlsx')
    dataset=dataset[0:200]
    dataset=dataset[['CuGds','IdIvcHdr']]
    data2=dataset[['CuGds','IdIvcHdr']]
    unique_arr = dataset["IdIvcHdr"].unique()
    unique_arr2=dataset["CuGds"].unique()
    #dataset.groupby("IdIvcHdr")
    
    data2.groupby(['IdIvcHdr', 'CuGds'])
    data2.to_csv('data2.csv')
    
   
    #dataset=pd.DataFrame(a,columns=unique_arr2)
    a=[]
    for i in unique_arr:
        
        #a.append(dataset.get_group(i)['IdIvcHdr'])
        a.append(dataset[dataset['IdIvcHdr']==i]['CuGds'])
        
    dataset=pd.DataFrame(a,index=unique_arr)    
    
    s=pd.DataFrame()
    
    '''p=[]

    for i in unique_arr:
   
      p.append(dataset[dataset['IdIvcHdr']==i]['CuGds'])
      
    dataset=pd.DataFrame(p,index=unique_arr,columns=None)'''
    
    number_records=len(dataset)
       
    return dataset,number_records
    
    
#json_ = request.json
json_=pd.read_excel('Goods In Invoices.xlsx')
json_=json_[0:300]
dataset,number_records=preprocess_dataset()
f=len(dataset.columns)
transactions = []
             
             
for i in range(0, number_records):
    
    transactions.append([str(dataset.values[i,j]) for j in range(0,f) if str(dataset.values[i,j]) != 'nan' ])
             
             
from apyori import apriori
Association_Rules = apriori(transactions, 
min_support = 0.005,
             min_confidence = 0.05,
min_lift = 2,
                min_length = 2,max_length=15)

Results = list(Association_Rules)
s=pd.DataFrame(transactions)
s.to_csv('transactions.csv')
             #print(len(Results))
             #print(Results)
#m=[]
#n=[]
#o=[]
rsh=[]
lsh=[]
support=[]
confedence=[]
lift=[]

for i in range(0,len(Results)):
    for j in range(0,len(Results[i][2])):
        rsh.append(list(Results[i][2][j][1]))
        support.append(Results[i][1])
        confedence.append(Results[i][2][j][2])
        lift.append(Results[i][2][j][3])
        lsh.append(list(Results[i][2][j][0]))
        
        
        
                   





'''for item in Results:

                 # first index of the inner list
                 # Contains base item and add item
                 pair = item[0]


                 pair=tuple(pair)
                 items = [x for x in pair]
                 a=items
                 b="Support: " + str(item[1])

                 #third index of the list located at 0th
                 #of the third index of the inner list

                 c="Confidence:" + str(item[2][0][2])
                 d="Lift: " + str(item[2][0][3])
                 e="====================================="
                 m.append(a)
                 n.append(b)
                 o.append(c)
                 #m.append(d)
                 #m.append(e)
                 
             #return  '{} \n{} '.format(m,e)'''

df=pd.DataFrame([(a, b,c,d,e) for a, b,c,d,e in zip(lsh,rsh,support,confedence,lift)],
                  columns=['گروه کالا1','گروه کالا2','support','ضریب اطمینان','lift'])

#df.to_csv('AprioryJob.csv')
df.to_csv(path+'/AprioryJob.csv')
print(Results)






def inspect(output):


    lhs = []
    rhs = []
    support = []
    confedence = []
    lift = []
    for i in range(0, len(output)):
        for j in range(0, len(output[i][2])):
            # lhs=[tuple(output[i][2][j])[0]]
            lhs.append(list(output[i][2][j][0]))
            # lhs.append( x for x in tuple(output[i][2][j])[0])

            # rhs.append(tuple(output[i][2][j])[1])
            rhs.append(list(output[i][2][j][1]))

            support.append(output[i][1])

            confedence.append(output[i][2][j][2])

            lift.append(output[i][2][j][3])

    return list(zip(lhs, rhs, support, confedence, lift))


output_dataframe = pd.DataFrame(inspect(Results),
                                columns=['Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confident', 'Lift'],
                                dtype="int")
output_dataframe = output_dataframe.sort_values('Lift', ascending=False)
# output_dataframe['Left_Hand_Side']=output_dataframe['Left_Hand_Side'].astype(int)
# output_dataframe['Right_Hand_Side']=output_dataframe['Right_Hand_Side'].astype(int)
#output_dataframe.to_csv(
   # 'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/' + str(CustomerName) + 'output.csv')
# r=type(output_dataframe.iloc[0:0])
m = []
j = []

for item in Results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]

    pair = tuple(pair)
    items = [x for x in pair]
    rule = items[0] + "---->" + "".join(items[1:])
    a = items
    b = "Support: " + str(item[1])

    # third index of the list located at 0th
    # of the third index of the inner list

    c = "Confidence:" + str(item[2][0][2])
    d = "Lift: " + str(item[2][0][3])
    e = "====================================="
    m.append(rule)
    # m.append(a)
    # m.append(b)
    # m.append(c)
    # m.append(d)
    # m.append(e)
    j.append(item)
v1 = list(output_dataframe.iloc[:, 0])
v2 = list(output_dataframe.iloc[:, 1])
m = pd.DataFrame(m)
#m.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/' + str(CustomerName) + 'm.csv')
v1 = list(output_dataframe.iloc[:, 0])
v2 = list(output_dataframe.iloc[:, 1])
confident = list(output_dataframe.iloc[:, 3])
lift = list(output_dataframe.iloc[:, 4])

conn = pyodbc.connect('Driver={SQL Server};Server=192.168.100.17\sql2019;Database=Modern_Master;uid=sa;pwd=PAYA+master')
cursor = conn.cursor()
cursor.execute("DELETE FROM  bi.GdsRuleId  ;")
cursor.execute("DELETE FROM  dbo.[bi.GdsRuleDetail1]  ;")
cursor.execute("DELETE FROM  dbo.[bi.GdsRuleDetail2]  ;")

conn.commit()
for i in range(0, len(v1)):
    cursor.execute("INSERT INTO  bi.GdsRuleId (GdsRuleId,lift,confident)  VALUES (?,?,?);", i, float(lift[i]),
                   float(confident[i]))
    conn.commit()

for i in range(0, len(v1)):
    for j in range(0, len(v1[i])):
        cursor.execute("INSERT INTO  dbo.[bi.GdsRuleDetail1] (IdHdr,IdGds)  VALUES (?,?);", float(i), float(v1[i][j]))
        conn.commit()
for i in range(0, len(v2)):
    for j in range(0, len(v2[i])):
        cursor.execute("INSERT INTO  dbo.[bi.GdsRuleDetail2] (IdHdr,IdGds)  VALUES (?,?);", float(i), float(v2[i][j]))
        conn.commit()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    