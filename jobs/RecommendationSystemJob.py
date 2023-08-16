# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:48:56 2022

@author: m.dorosti
"""

import pandas as pd
import numpy as np
import pyodbc
from timeit import default_timer as timer
conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=BarshicData_2;'
                                  'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )
               
               
with open('C:\\Users\\m.dorosti\\Desktop\\RecommenderBinaryCF-main\\SalesDetail.txt' ,'r')as file:
                   query=file.read()
                   
                   
cursor = conn.cursor()   

import pandas as pd
GF=pd.read_sql_query(query,conn) 
                                
data=GF
DataPrep = data[['DscGds', 'SalesAmount', 'IdPrsClient']]
DataPrep.rename(columns={'DscGds':'SalesItem','IdPrsClient':'Customer'},inplace=True)
DataGrouped = DataPrep.groupby(['Customer', 'SalesItem']).sum().reset_index()
customers = list(np.sort(DataGrouped.Customer.unique())) # why 36 unique customers in a list and not 35? Index starts at 0!
products = list(DataGrouped.SalesItem.unique()) # Get our unique 3725 unique products that were purchased
quantity = list(DataGrouped.SalesAmount) # All of our purchases
#list function is a list of values. So customers now stores 36 unique customers.
from pandas import DataFrame
DfCustomerUnique = DataFrame(customers,columns=['Customer'])
from scipy import sparse
from pandas.api.types import CategoricalDtype

rows = DataGrouped.Customer.astype(CategoricalDtype(categories=customers)).cat.codes # We have got 36 unique customers, which make up 13837 data rows (index)

# Get the associated row indices
cols = DataGrouped.SalesItem.astype(CategoricalDtype(categories= products)).cat.codes # We have got unique 3725 SalesItems, making up 13837 data rows (index)

# Get the associated column indices
#Compressed Sparse Row matrix
PurchaseSparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products))) #len of customers=35, len of products=3725
#csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
#where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k]. , see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
#We have 35 customers with 3725 items. For these user/item interactions, 13837 of these items had a purchase. 
#In terms of sparsity of the matrix, that makes:
MatrixSize = PurchaseSparse.shape[0]*PurchaseSparse.shape[1] # 130375 possible interactions in the matrix (35 unique customers * 3725 unique SalesItems=130375)
PurchaseAmount = len(PurchaseSparse.nonzero()[0]) # 13837 SalesItems interacted with; 
sparsity = 100*(1 - (PurchaseAmount/MatrixSize))

def create_DataBinary(DataGrouped):
  # DataPrep must be DataGrouped?!
    DataBinary = DataPrep.copy()
    DataBinary['PurchasedYes'] = 1 
    return DataBinary

DataBinary = create_DataBinary(DataGrouped)
data2=DataBinary.drop(['SalesAmount'], axis=1)
data2['SalesItem']=data2['SalesItem'].astype(str)
DfMatrix = pd.pivot_table(data2, values='PurchasedYes', index='Customer', columns='SalesItem')
DfMatrix=DfMatrix.fillna(0)
DfMatrixNorm3=(DfMatrix-DfMatrix.min())/(DfMatrix.max()-DfMatrix.min())
DfResetted = DfMatrix.reset_index().rename_axis(None, axis=1) 
df=DfResetted
df=df.fillna(0)
df.head()
DfSalesItem=df.drop('Customer',1)
DfSalesItemNorm = DfSalesItem / np.sqrt(np.square(DfSalesItem).sum(axis=0)) 
# Calculating with Vectors to compute Cosine Similarities
ItemItemSim = DfSalesItemNorm.transpose().dot(DfSalesItemNorm) 
ItemNeighbours = pd.DataFrame(index=ItemItemSim.columns,columns=range(1,10))
for i in range(0,len(ItemItemSim.columns)):
    ItemNeighbours.iloc[i,:9] = ItemItemSim.iloc[0:,i].sort_values(ascending=False)[:9].index
ItemNeighbours.to_excel("1049738ExportItem-Item-data_neighbours.xlsx")
#Now we will build a Customer based recommendation, which is build upon the item-item similarity matrix, which we have just calculated above.
# Create a place holder matrix for similarities, and fill in the customer column
CustItemSimilarity = pd.DataFrame(index=df.index,columns=df.columns)
CustItemSimilarity.iloc[:,:1] = df.iloc[:,:1]
# Getting the similarity scores
def getScore(history, similarities):
   return sum(history*similarities)/(sum(similarities)+0.0001) 
start = timer()
for i in range(0,len(CustItemSimilarity.index)):
    for j in range(1,len(CustItemSimilarity.columns)):
        user = CustItemSimilarity.index[i]
        product = CustItemSimilarity.columns[j]
 
        if df.loc[i][j] == 1:
            CustItemSimilarity.loc[i][j] = 0
        else:
            ItemTop = ItemNeighbours.loc[product][1:9] #
            #do not use order but sort_values in latest pandas
            ItemTopSimilarity = ItemItemSim.loc[product].sort_values(ascending=False)[1:9]
            #here we will use the item dataframe, which we generated during item-item matrix 
            CustomerPurchasings = DfSalesItem.loc[user,ItemTop]
 
            CustItemSimilarity.loc[i][j] = getScore(CustomerPurchasings,ItemTopSimilarity)

end = timer()

print('\nRuntime: %0.2fs' % (end - start))

ItemTop = ItemNeighbours.loc[product][1:9]
#now generate a matrix of customer based recommendations
CustItemRecommend = pd.DataFrame(index=CustItemSimilarity.index, columns=['Customer','item1','item2','item3','item4','item5','item6']) #Top 1,2..6
CustItemRecommend.iloc[0:,0] = CustItemSimilarity.iloc[:,0]
#Instead of having the matrix filled with similarity scores we want to see the product names.
for i in range(0,len(CustItemSimilarity.index)):
    CustItemRecommend.iloc[i,1:] = CustItemSimilarity.iloc[i,:].sort_values(ascending=False).iloc[1:7,].index.transpose()
CustItemRecommend.to_excel("1049738ExportCustomer-Item-CustItemRecommend.xlsx")    

    



























