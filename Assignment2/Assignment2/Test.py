#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas  as pd
import numpy as np
import pickle
from Training import extract_features
from sklearn.preprocessing import MinMaxScaler

raw_data_matrix = pd.read_csv("test.csv")
feature_matrix = pd.DataFrame(columns = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11'])
for i in range(raw_data_matrix.shape[0]):
    feature_matrix.loc[i]=extract_features(raw_data_matrix.loc[i].tolist())

scaler = pickle.load(open("scaler.pkl", 'rb'))
feature_matrix = scaler.transform(feature_matrix)
loaded_model = pickle.load(open("Model.pkl", 'rb'))

pred = loaded_model.predict(feature_matrix)
outfile=open('Result.csv','w')
for value in pred:
    outfile.write(str(value)+'\n')
outfile.close()


# In[ ]:





# In[ ]:




