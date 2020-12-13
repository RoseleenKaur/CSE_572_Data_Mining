#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
import math

def create_meal_nomeal_date_list(InsulinData_timestamp_df,CGMData):
    InsulinData_timestamp=InsulinData_timestamp_df['DateTime'].tolist()
    InsulinData_true_label=InsulinData_timestamp_df['Ground Truth Label'].tolist()
    size=len(InsulinData_timestamp)
    meal_date_df=pd.DataFrame(columns = ['DateTime','Ground Truth Label'])
    no_meal_date_list=[]
    #no_meal_date_list=[]
    for i in range(size):
        index=(size-1)-i
    
        date=CGMData[CGMData['DateTime']>=InsulinData_timestamp[index]]['DateTime'].min()
        label=InsulinData_true_label[index]
        #date=InsulinData_timestamp[index]
        next_date=pd.to_datetime(str(date.date())+' '+'23:59:59', format='%Y/%m/%d %H:%M:%S') if(index==0) else InsulinData_timestamp[index-1]
        # print(""+str(next_date)+"|"+ str(date)+"|"+str(next_date-date))
        days=(next_date-date).days
        hrs=(next_date-date).seconds//3600
    
        if(days>0 or hrs>2):
            #print(days,hrs)
            #meal_date_list.append(date)
            #meal_date_df.append(date,label)
            var = {'DateTime':date, 'Ground Truth Label': label} 
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
    
    #Feature1
    CGMmax=max(raw_data_vector)
    CGMmin=min(raw_data_vector)
    feature_vector.append(CGMmax-CGMmin)
    
    
    #Feature2
    CGMmax=max(raw_data_vector)
    max_index=raw_data_vector.index(CGMmax)
    min_index= 0 if max_index==0 else raw_data_vector.index(min(raw_data_vector[0:max_index]))
    y_value = raw_data_vector[min_index] + 0.1 * (raw_data_vector[max_index] - raw_data_vector[min_index])

    count=0
    x_first=0
    for i in range(min_index,max_index):
        if(y_value-raw_data_vector[i])>=0 :
            x_first=i

    x_secound=0
    for i in range(len(raw_data_vector)-1,max_index,-1):
        if(y_value-raw_data_vector[i])>=0 :
            x_secound=i
    feature_vector.append(x_secound - x_first)
    
    
    
    
    #Feature3 Velocity
    velocity_list=[]
    for i in range(1,len(raw_data_vector)):
        velocity_list.append(raw_data_vector[i]-raw_data_vector[i-1])
    max_vel=max(velocity_list)
    max_vel_id=velocity_list.index(max_vel)
    feature_vector.extend([max_vel,max_vel_id])
    feature_vector.extend([sum(velocity_list)])
    
    #Feature4 Acceleration
    velocity_list=[]
    for i in range(1,len(raw_data_vector)):
        velocity_list.append(raw_data_vector[i]-raw_data_vector[i-1])
    acc_list=[]
    for i in range(1,len(velocity_list)):
        acc_list.append(velocity_list[i]-velocity_list[i-1])
    
    max_acc=max(acc_list)
    max_acc_id=acc_list.index(max_acc)
    feature_vector.extend([max_acc,max_acc_id])
    feature_vector.extend([sum(acc_list)])
    
    #Feature5
    rfft = np.fft.rfft(raw_data_vector)
    rfft_log = np.log(np.abs(rfft) ** 2 + 1)
    feature_vector.extend(rfft_log[7:10])
    
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
        feature_matrix.loc[matrix_length+i]=arr+[meal_date_df.iloc[i]['Ground Truth Label']]
        
def merge_cluster(cluster_index,cls):
    df_cls=pd.DataFrame(cls)
    dicti={}
    for i in range(max(cls)+1):
        dicti[i]=df_cls[df_cls[0]==i].index.tolist()
    nearest_clus=find_nearest_cluster(cluster_index,dicti)
    #print(nearest_clus)
    df_cls.loc[df_cls[0]==cluster_index,[0]]=nearest_clus
    df_cls.loc[df_cls[0]>cluster_index,[0]]-=1
    
    return df_cls

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
    InsulinData=InsulinData[['DateTime',"BWZ Carb Input (grams)"]]
    InsulinData=InsulinData.dropna()
    InsulinData.drop(InsulinData[InsulinData["BWZ Carb Input (grams)"]==0].index, inplace = True)
    InsulinData=InsulinData.reset_index(drop = True)
    
    #calculating number of clusters
    min_val=InsulinData["BWZ Carb Input (grams)"].min()
    max_val=InsulinData["BWZ Carb Input (grams)"].max()
    num_of_clusters=math.ceil((max_val-min_val)/20)
    
    #labeling ground truth
    InsulinData['Ground Truth Label']=((InsulinData["BWZ Carb Input (grams)"]-min_val+1)/20).apply(np.ceil)
    
    #loading CGMData
    CGMData=pd.read_csv("CGMData.csv")
    CGMData['DateTime']= pd.to_datetime(CGMData['Date']+' '+CGMData['Time'], format='%m/%d/%Y %H:%M:%S')
    CGMData=CGMData[['DateTime','Date','Time',"Sensor Glucose (mg/dL)"]] 
    CGMData["Sensor Glucose (mg/dL)"]=CGMData["Sensor Glucose (mg/dL)"].interpolate(method ='linear', limit_direction ='both')
    
    meal_date_df=create_meal_nomeal_date_list(InsulinData,CGMData)
    
    feature_matrix = pd.DataFrame(columns = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11','Class Label'])
    create_feature_matrix(meal_date_df,CGMData,feature_matrix)
    

    feature_matrix.drop_duplicates(subset =['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11'],keep=False,inplace=True)
    feature_matrix=feature_matrix.reset_index(drop = True) 

    X = feature_matrix[['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11']]
    y = feature_matrix[['Class Label']]

    X_normalized = MinMaxScaler().fit(X).transform(X)  

    #applying K-means
    kmeans = KMeans(n_clusters=num_of_clusters,random_state=0)
    kmeans.fit(X_normalized)
    feature_matrix['K-means Label']=kmeans.labels_
    feature_matrix['Class Label']=feature_matrix['Class Label']-1
    kmeans_sse=kmeans.inertia_
    
    cluster_distribution = pd.DataFrame(columns = ['Bin 0','Bin 1','Bin 2','Bin 3','Bin 4','Bin 5','Bin 6'])
    for cluster in range(0,7):
        for bin in range(0,7):
            val=feature_matrix[(feature_matrix['Class Label']==bin) &(feature_matrix['K-means Label']==cluster)].shape[0]
            cluster_distribution.at[cluster,'Bin '+str(bin)]= val
            
    entropy=np.zeros(7);
    purity=np.zeros(7);
    for i in range(0,7):
        s=cluster_distribution.iloc[i].sum()
        maximum=cluster_distribution.iloc[i].max()
        entropy[i]=(np.log2(cluster_distribution.iloc[i].astype('float')/s)*((cluster_distribution.iloc[i]/s)*-1)).sum()
        purity[i] = maximum/s;
    
    wholeEntropy = 0
    wholePurity = 0
    cluster_total=cluster_distribution.sum(axis=1)
    total_sum=cluster_distribution.sum(axis=1).sum()
    for i in range(0,7):
        wholeEntropy+=(cluster_total[i]/total_sum)*entropy[i]
        wholePurity+=(cluster_total[i]/total_sum)*purity[i]
    kmeans_entropy=wholeEntropy
    kmeans_purity=wholePurity
    
    #determining eps value for dbscan
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X_normalized)
    distances, indices = nbrs.kneighbors(X_normalized)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances);
    
    
    ##DBSCAN and its metrics
    dbscan = DBSCAN(eps = 0.3, min_samples = 2)
    cls = dbscan.fit_predict(X_normalized)
    #print("Cluster membership values:\n{}".format(cls))


    new_X=pd.DataFrame(X_normalized)
    dist_matrix=pd.DataFrame(distance_matrix(new_X.values, new_X.values), index=new_X.index, columns=new_X.index)

    num_of_cluster = max(cls)

    #merging extra clusters
    for i in range(num_of_cluster-6):
        count_arr=np.array(np.unique(cls, return_counts=True)).T
        count_dict={}
        for i in range(count_arr.shape[0]):
            count_dict[count_arr[i][0]]=count_arr[i][1]
        temp = min(count_dict.values()) 
        smallest_cluster = [key for key in count_dict if count_dict[key] == temp] 
        df_cls=merge_cluster(smallest_cluster[-1],cls)
        cls=df_cls[0].tolist()

 
    #merging noise points
    df_cls=pd.DataFrame(cls)
    noise_dict={}
    for i in range(-1,max(cls)+1):
        noise_dict[i]=df_cls[df_cls[0]==i].index.tolist()

    for i in noise_dict[-1]:
        temp_dict=noise_dict.copy()
        temp_dict[-1]=list([i])
        df_cls.loc[df_cls.index==i,[0]]=find_nearest_cluster(-1,temp_dict)
    
    cls=df_cls[0].tolist()
    
    feature_matrix['DBSCAN Label']=cls
    
    cluster_distribution = pd.DataFrame(columns = ['Bin 0','Bin 1','Bin 2','Bin 3','Bin 4','Bin 5','Bin 6'])
    for cluster in range(0,7):
        for bin in range(0,7):
            val=feature_matrix[(feature_matrix['Class Label']==bin) &(feature_matrix['DBSCAN Label']==cluster)].shape[0]
            cluster_distribution.at[cluster,'Bin '+str(bin)]= val
    entropy=np.zeros(7);
    purity=np.zeros(7);
    for i in range(0,7):
        s=cluster_distribution.iloc[i].sum()
        if(s==0):
            continue
        maximum=cluster_distribution.iloc[i].max()
        entropy[i]=(np.log2(cluster_distribution.iloc[i].astype('float')/s)*((cluster_distribution.iloc[i]/s)*-1)).sum()
        purity[i] = maximum/s;
    
    wholeEntropy = 0
    wholePurity = 0
    cluster_total=cluster_distribution.sum(axis=1)
    total_sum=cluster_distribution.sum(axis=1).sum()
    for i in range(0,7):
        wholeEntropy+=(cluster_total[i]/total_sum)*entropy[i]
        wholePurity+=(cluster_total[i]/total_sum)*purity[i]
    dbscan_entropy=wholeEntropy
    dbscan_purity=wholePurity

    dbscan_sse=0
    X_normalized_df=pd.DataFrame(X_normalized)
    X_normalized_df['DBSCAN Label']=feature_matrix['DBSCAN Label']
    for cluster in range(0,7):
        temp=X_normalized_df[(X_normalized_df['DBSCAN Label']==cluster)]
        mean=np.array(temp.iloc[:, 0:11].mean())
        for i in range(temp.shape[0]):
            point=np.array(temp.iloc[i,0:11])
            diff=point-mean
            dbscan_sse+=np.dot(diff.T, diff)
            
    #creating output file/results    
    answer_df=pd.DataFrame()
    answer_df['SSE for Kmeans']=[kmeans_sse]
    answer_df['SSE for DBSCAN']=[dbscan_sse]
    answer_df['Entropy for Kmeans']=[kmeans_entropy]
    answer_df['Entropy for DBSCAN']=[dbscan_entropy]
    answer_df['Purity for Kmeans']=[kmeans_purity]
    answer_df['Purity for DBSCAN']=[dbscan_purity]
    answer_df.to_csv('Results.csv')


# In[ ]:




