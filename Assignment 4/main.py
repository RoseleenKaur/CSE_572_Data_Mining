#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import math

def create_meal_nomeal_date_list(InsulinData_timestamp_df,CGMData):
    InsulinData_timestamp=InsulinData_timestamp_df['DateTime'].tolist()
    InsulinData_bolus_insulin=InsulinData_timestamp_df['BWZ Estimate (U)'].tolist()
    size=len(InsulinData_timestamp)
    meal_date_df=pd.DataFrame(columns = ['DateTime','BWZ Estimate (U)'])
    no_meal_date_list=[]
    #no_meal_date_list=[]
    for i in range(size):
        index=(size-1)-i
    
        date=CGMData[CGMData['DateTime']>=InsulinData_timestamp[index]]['DateTime'].min()
        label=InsulinData_bolus_insulin[index]
        #date=InsulinData_timestamp[index]
        next_date=pd.to_datetime(str(date.date())+' '+'23:59:59', format='%Y/%m/%d %H:%M:%S') if(index==0) else InsulinData_timestamp[index-1]
        # print(""+str(next_date)+"|"+ str(date)+"|"+str(next_date-date))
        days=(next_date-date).days
        hrs=(next_date-date).seconds//3600
    
        if(days>0 or hrs>2):
            #print(days,hrs)
            #meal_date_list.append(date)
            #meal_date_df.append(date,label)
            var = {'DateTime':date, 'BWZ Estimate (U)': label} 
            meal_date_df = meal_date_df.append(var, ignore_index = True) 
            '''
            no_meal_starttime=date+pd.to_timedelta(2, unit='h')
            no_meal_starttime=CGMData[CGMData['DateTime']>=no_meal_starttime]['DateTime'].min()
            while((next_date-no_meal_starttime).days>0 or (next_date-no_meal_starttime).seconds//3600>=2):
                no_meal_date_list.append(no_meal_starttime)
                no_meal_starttime+=pd.to_timedelta(2, unit='h')
            '''
    return (meal_date_df)

def extract_features(raw_data_vector):
    feature_vector=[]
    
    CGMmax=max(raw_data_vector)
    CGMmin=min(raw_data_vector)
    
    
    #CGMMeal
    feature_vector.append(raw_data_vector[0])
    
    #CGMMax
    feature_vector.append(CGMmax)
    
    #CGMMin
    feature_vector.append(CGMmin)
    
    return feature_vector
    

def create_feature_matrix(meal_date_df,CGMData,feature_matrix):   
    matrix_length=feature_matrix.shape[0]    
    for i in range(meal_date_df.shape[0]):
        meal_date=meal_date_df.iloc[i]['DateTime']
        arr=CGMData[(CGMData['DateTime']>=(meal_date-pd.to_timedelta(30, unit='m'))) & (CGMData['DateTime']<(meal_date+pd.to_timedelta(2, unit='h')))]["Sensor Glucose (mg/dL)"].to_list()
        if(len(arr)<24):
            continue;
        arr.reverse()  
        arr=extract_features(arr)
        feature_matrix.loc[matrix_length+i]=arr+[meal_date_df.iloc[i]['BWZ Estimate (U)']]
        


def find_nearest_cluster(cluster_index,dt):
    dist_arr=[]
    min_cluster_id=0;
    min_cluster_dist=math.inf
    for i in range(0,len(dt)):
        if(i not in dt):
            continue;
        if cluster_index==i:
            dist_arr.append(0)
            continue
        else:
            min_val=math.inf
            for k in dt[cluster_index]:
                for j in dt[i]:
                    #print(dt)
                    if dist_matrix[k][j]<min_val:
                        min_val=dist_matrix[k][j]
        dist_arr.append(min_val)
        if(min_val<min_cluster_dist):
            min_cluster_dist=min_val
            min_cluster_id=i
    return min_cluster_id;    
    
        
if __name__ == '__main__':
    
    #Loading Insulin data
    InsulinData=pd.read_csv("InsulinData.csv")
    InsulinData['DateTime']= pd.to_datetime(InsulinData['Date']+' '+InsulinData['Time'], format='%m/%d/%Y %H:%M:%S')
    InsulinData=InsulinData[['DateTime',"BWZ Carb Input (grams)",'BWZ Estimate (U)']]
    InsulinData=InsulinData.dropna()
    InsulinData.drop(InsulinData[InsulinData["BWZ Carb Input (grams)"]==0].index, inplace = True)
    InsulinData=InsulinData.reset_index(drop = True)
    
    #calculating number of clusters
    min_val=InsulinData["BWZ Carb Input (grams)"].min()
    max_val=InsulinData["BWZ Carb Input (grams)"].max()
    num_of_clusters=math.ceil((max_val-min_val)/20)
    
    
    #loading CGMData
    CGMData=pd.read_csv("CGMData.csv")
    CGMData['DateTime']= pd.to_datetime(CGMData['Date']+' '+CGMData['Time'], format='%m/%d/%Y %H:%M:%S')
    CGMData=CGMData[['DateTime','Date','Time',"Sensor Glucose (mg/dL)"]] 
    CGMData["Sensor Glucose (mg/dL)"]=CGMData["Sensor Glucose (mg/dL)"].interpolate(method ='linear', limit_direction ='both')
    
    meal_date_df=create_meal_nomeal_date_list(InsulinData,CGMData)
    
    feature_matrix = pd.DataFrame(columns = ['CGMMeal','CGMMax','CGMMin','I_bolus'])
    create_feature_matrix(meal_date_df,CGMData,feature_matrix)
    
    feature_matrix=feature_matrix.reset_index(drop = True) 
    
    #calculating number of bins
    global_CGM_min=feature_matrix["CGMMin"].min()
    global_CGM_max=feature_matrix["CGMMax"].max()
    num_of_bins=math.ceil((global_CGM_max-global_CGM_min+1)/20)
    
    #labeling bin number for CGMMax and CGMMeal
    feature_matrix['B_max']=((feature_matrix['CGMMax']-global_CGM_min)/20).apply(np.floor)
    feature_matrix['B_meal']=((feature_matrix['CGMMeal']-global_CGM_min)/20).apply(np.floor)
    
    #rounding off I_bolus value
    feature_matrix['I_bolus']=feature_matrix['I_bolus'].round(0)
    
    #Association Rule mining
    feature_matrix["I_bolus"]=pd.to_numeric(feature_matrix["I_bolus"], downcast='integer')
    Itemsets=feature_matrix[['B_max','B_meal','I_bolus']]
    Itemsets_freq=Itemsets.groupby(['B_max','B_meal','I_bolus']).size().reset_index(name='rule_count')
    Itemsets_freq=Itemsets_freq.sort_values(by = 'rule_count',ascending = False) 
    Itemsets_freq.head(1)[['B_max','B_meal','I_bolus']].to_csv('most_frequent_itemsets.csv',index=False) 
    
    Itemsets_X_freq=Itemsets.groupby(['B_max','B_meal']).size().reset_index(name='X_count')
    Rule_confidence=pd.merge(Itemsets_freq, Itemsets_X_freq, how ='inner', on =['B_max','B_meal'])
    Rule_confidence['confidence']=Rule_confidence['rule_count']/Rule_confidence['X_count']
    Rule_confidence[Rule_confidence['confidence']==1][['B_max','B_meal','I_bolus']].to_csv('largest_confidence_rules.csv',index=False)
    Rule_confidence[Rule_confidence['confidence']<0.15].sort_values(by = 'confidence')[['B_max','B_meal','I_bolus']].to_csv('less_than_15_percent_confidence_rules.csv',index=False)

   

