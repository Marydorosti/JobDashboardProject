# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:56:22 2021

@author: m.dorosti
"""
# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import sys
import json
from flask import Flask, request, render_template, session, redirect
from sklearn.cluster import KMeans


from flask import Blueprint
kmeansApp_bp = Blueprint('kmeansApp', __name__)




# Your API definition
#app = Flask(__name__,template_folder='template')




def preprocess_data(json_):
               import pandas as pd
               import datetime as dt
              
               
               data=pd.DataFrame(json_)
              

               
               
               
               
           
            
            
            
                #import modules
               import pandas as pd # for dataframes
                
               import datetime as dt
                
               data.head()


                # Z score
               from scipy import stats
               #import numpy as np
  
               z_score = np.abs(stats.zscore(data['SumPrice']))


                #threshold = 10
  
                # Position of the outlier
               a=np.where(z_score > 1000)
 

    
               #print("Old Shape: ", data.shape) 
               #data.drop(np.where(z > 4), inplace = True)
               data.drop(a[0], inplace = True)
               #print("New Shape: ", data.shape)
               #PRESENT = dt.datetime(2021,6,20)
               PRESENT = dt.datetime.today()
               #data[['Dt_Effect']]=int(data[['Dt_Effect']])
               data['Dt_Effect'] = pd.to_datetime(data['Dt_Effect'])
               

               #rfm= data.groupby('IdPrsClient').agg({'Dt_Effect': lambda date: (PRESENT - date.max()).days,
                                        #'CountIvc': lambda num: len(num),
                                        #'SumPrice': lambda price: price.sum()})
               rfm= data.groupby('IdPrsClient').agg({'Dt_Effect': lambda date: (PRESENT - date.max()).days,
                                        'CountIvc': lambda num: num.sum(),
                                        'SumPrice': lambda price: price.sum()})
  
  
               rfm.columns=['recency','frequency','monetary']
  
               rfm['recency'] = rfm['recency'].astype(int)


               rfm['r_quartile'] = pd.qcut(rfm['recency'],4, ['1','2','3','4'], duplicates='drop')
               rfm['f_quartile'] = pd.qcut(rfm['frequency'],4, ['4','3','2','1'], duplicates='drop')
               rfm['m_quartile'] = pd.qcut(rfm['monetary'],4, ['4','3','2','1'])

               rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
               #rfm['RFM_Score'] = rfm.r_quartile.astype(str)+rfm.m_quartile.astype(str)
               #rfm['RFM_Score'] =  rfm.m_quartile.astype(str)
            
               df=rfm[['RFM_Score']]
                
               quotient_2d = np.array(df).reshape(-1,1)
            
               return quotient_2d,df
    

















@kmeansApp_bp.route('/kmeansApp',methods=['POST'])
#@app.route('/predict', methods=['POST'])
def kmeansApp():
    #if k_means:
        try:
               json_ = request.json
               #a_json = json.loads(json_)
               import joblib
               import traceback
               import pandas as pd
               import datetime
               import numpy as np
               import datetime as dt
              
               import sys
               quotient_2d,df=preprocess_data(json_)
               '''k_means = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
               n_clusters=4, n_init=10,  precompute_distances='auto',
               random_state=0, tol=0.0001, verbose=0)'''
               k_means=KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
               k_means.fit(quotient_2d)

               z = k_means.predict(quotient_2d)
               cluster_map = pd.DataFrame()
               cluster_map['data_index'] = df.index.values
               cluster_map['cluster'] =z
               cluster_map['customer grade']=cluster_map['cluster'].replace({0:"d",1:"a",2:"c",3:"b"})
               
               #def html_table():

               #return render_template('simple.html.html',  tables=[cluster_map.to_html(classes='data')], titles=df.columns.values)
               #return jsonify({"data_index":df.index.values ,"cluster":z,"cluster_map":cluster_map['cluster'].replace({0:"d",1:"a",2:"c",3:"b"}) })
               #return jsonify(cluster_map)
               #dfList = cluster_map.values.tolist()
               #return jsonify(dfList)
               return cluster_map.to_json(orient="records")


            
 
             

               #from flask_jsonpify import jsonpify
           
               #return (str(cluster_map))

        except:

             return jsonify({'trace': traceback.format_exc()})
    #else:
        #print ('Train the model first')
        #return ('No model here to use')

'''if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 14444 # If you don't provide any port the port will be set to 12345

    #XG = joblib.load("model.pkl") # Load "model.pkl"
    #k_means = joblib.load('model.pkl') 
    #datapreprocess=joblib.load('function.pkl')
    #print ('Model loaded')
    #model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    #print ('Model columns loaded')

    app.run(port=port, debug=True)'''
