#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import random 
import math
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import pickle

def load_InsulinData(filename,patient):
    if patient==1:
        InsulinData=pd.read_csv("InsulinData.csv")
        InsulinData['DateTime']= pd.to_datetime(InsulinData['Date']+' '+InsulinData['Time'], format='%m/%d/%Y %H:%M:%S')
    else:
        InsulinData=pd.read_excel("InsulinAndMealIntake670GPatient3.xlsx")
        InsulinData['DateTime']= pd.to_datetime(InsulinData['Date'].astype(str).str.cat(InsulinData['Time'].astype(str),sep=" "), format='%Y-%m-%d %H:%M:%S')

    InsulinData=InsulinData[['DateTime',"BWZ Carb Input (grams)"]]
    InsulinData=InsulinData.dropna()
    InsulinData.drop(InsulinData[InsulinData["BWZ Carb Input (grams)"]==0].index, inplace = True)
    InsulinData=InsulinData.reset_index(drop = True) 

    return InsulinData['DateTime'].tolist()


def load_CGMData(filename,patient):
    if patient==1:
        CGMData=pd.read_csv(filename)
        CGMData['DateTime']= pd.to_datetime(CGMData['Date']+' '+CGMData['Time'], format='%m/%d/%Y %H:%M:%S')
    else:
        CGMData=pd.read_excel(filename)
        CGMData['DateTime']= pd.to_datetime(CGMData['Date'].astype(str).str.cat(CGMData['Time'].astype(str),sep=" "), format='%Y/%m/%d %H:%M:%S')
    CGMData=CGMData[['DateTime','Date','Time',"Sensor Glucose (mg/dL)"]] 
    CGMData["Sensor Glucose (mg/dL)"]=CGMData["Sensor Glucose (mg/dL)"].interpolate(method ='linear', limit_direction ='both') 
    return CGMData


def create_meal_nomeal_date_list(InsulinData_timestamp,CGMData):
    size=len(InsulinData_timestamp)
    meal_date_list=[]
    no_meal_date_list=[]
    for i in range(size):
        index=(size-1)-i
    
        date=CGMData[CGMData['DateTime']>=InsulinData_timestamp[index]]['DateTime'].min()
        #date=InsulinData_timestamp[index]
        next_date=pd.to_datetime(str(date.date())+' '+'23:59:59', format='%Y/%m/%d %H:%M:%S') if(index==0) else InsulinData_timestamp[index-1]
        # print(""+str(next_date)+"|"+ str(date)+"|"+str(next_date-date))
        days=(next_date-date).days
        hrs=(next_date-date).seconds//3600
    
        if(days>0 or hrs>2):
            #print(days,hrs)
            meal_date_list.append(date)
            no_meal_starttime=date+pd.to_timedelta(2, unit='h')
            no_meal_starttime=CGMData[CGMData['DateTime']>=no_meal_starttime]['DateTime'].min()
            while((next_date-no_meal_starttime).days>0 or (next_date-no_meal_starttime).seconds//3600>=2):
                no_meal_date_list.append(no_meal_starttime)
                no_meal_starttime+=pd.to_timedelta(2, unit='h')
    return (meal_date_list,no_meal_date_list)
    
def extract_features(raw_data_vector):
    feature_vector=[]
    #Feature1
    CGMmax=max(raw_data_vector)
    CGMmin=min(raw_data_vector)
    feature_vector.append(CGMmax-CGMmin)
    
    #Feature2
    max_index=raw_data_vector.index(CGMmax)
    min_index= 0 if max_index==0 else raw_data_vector.index(min(raw_data_vector[0:max_index]))
    feature_vector.append(max_index-min_index)
    
    #Feature3
    velocity_list=[]
    for i in range(1,len(raw_data_vector)):
        velocity_list.append(raw_data_vector[i]-raw_data_vector[i-1])
    max_vel=max(velocity_list)
    max_vel_id=velocity_list.index(max_vel)
    feature_vector.extend([max_vel,max_vel_id])
    
    #Feature4
    binsize=len(raw_data_vector)/8
    start=0
    avg=[]
    for i in range(1,9):    
        end=math.floor(i*binsize)
        avg.append(sum(raw_data_vector[start:end])/(end-start))
        start=end
    feature_vector.extend(avg[1:6])
    
    #Feature5
    rfft = np.fft.rfft(raw_data_vector)
    rfft_log = np.log(np.abs(rfft) ** 2 + 1)
    feature_vector.extend(rfft_log[1:3])

    return feature_vector;
    

def create_feature_matrix(meal_date_list,no_meal_date_list,CGMData,feature_matrix):
    
    matrix_length=feature_matrix.shape[0]    
    for i in range(len(meal_date_list)):
        meal_date=meal_date_list[i]
        arr=CGMData[(CGMData['DateTime']>=(meal_date-pd.to_timedelta(30, unit='m'))) & (CGMData['DateTime']<(meal_date+pd.to_timedelta(2, unit='h')))]["Sensor Glucose (mg/dL)"].to_list()
        if(len(arr)<24):
            continue;
        arr.reverse()  
        arr=extract_features(arr)
        feature_matrix.loc[matrix_length+i]=arr+[1]

    matrix_length=feature_matrix.shape[0]
    
    for i in range(len(no_meal_date_list)):
        length=matrix_length
        no_meal_date=no_meal_date_list[i]
        arr=CGMData[(CGMData['DateTime']>=no_meal_date) & (CGMData['DateTime']<(no_meal_date+pd.to_timedelta(2, unit='h')))]["Sensor Glucose (mg/dL)"].to_list()
        if(len(arr)<24):
            continue;
        arr.reverse()
        arr=extract_features(arr)
        feature_matrix.loc[matrix_length+i]=arr+[0]
        

def k_fold_VAlidation(feature_matrix):
    scaler = MinMaxScaler()
    clf = SVC(kernel = 'rbf',gamma = 1,C = 150)
    X = feature_matrix[['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11']]
    y = feature_matrix['Class Label']
    X_scaled = scaler.fit_transform(X)
    cv_scores = cross_val_score(clf, X_scaled, y)
    print('Cross-validation scores (5-fold):', cv_scores)
    print('Mean cross-validation score (5-fold): {:.3f}'.format(np.mean(cv_scores)))

    
    
def training_machine(training_matrix,testing_matrix):
    scaler = MinMaxScaler()
    #X_train, X_test, y_train, y_test = train_test_split(feature_matrix[['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11']], feature_matrix['Class Label'])

    X_train= training_matrix[['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11']]
    y_train= training_matrix['Class Label']
    
    X_test= testing_matrix[['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11']]
    y_test= testing_matrix['Class Label']

    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = SVC(kernel = 'rbf', gamma = 1,C = 150).fit(X_train_scaled, y_train)

    print('Accuracy of RBF-kernel SVC on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
    print('Accuracy of RBF-kernel SVC on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))
  
    #Saving model
    filename = 'Model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    pickle.dump(scaler, open("scaler.pkl", 'wb'))
    


if __name__ == '__main__':
    #Patient 1 files are read_csv(), and Patient 2 files are read_excel()
    InsulinData_timestamp1=load_InsulinData("InsulinData.csv",1)
    InsulinData_timestamp2=load_InsulinData("InsulinAndMealIntake670GPatient3.xlsx",2)
    CGM_Data1=load_CGMData("CGMData.csv",1)
    CGM_Data2=load_CGMData("CGMData670GPatient3.xlsx",2)
    meal_date_list1,no_meal_date_list1=create_meal_nomeal_date_list(InsulinData_timestamp1,CGM_Data1)
    meal_date_list2,no_meal_date_list2=create_meal_nomeal_date_list(InsulinData_timestamp2,CGM_Data2)

    no_meal_date_list1=random.sample(no_meal_date_list1, min(2*len(meal_date_list1),len(no_meal_date_list1)))
    no_meal_date_list2=random.sample(no_meal_date_list2, min(2*len(meal_date_list2),len(no_meal_date_list2)))
    
    training_matrix = pd.DataFrame(columns = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11','Class Label'])
    testing_matrix = pd.DataFrame(columns = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Feature 11','Class Label'])
    
    create_feature_matrix(meal_date_list1,no_meal_date_list1,CGM_Data1,training_matrix)
    create_feature_matrix(meal_date_list2,no_meal_date_list2,CGM_Data2,testing_matrix)
    
    k_fold_VAlidation(training_matrix)
    training_machine(training_matrix,testing_matrix)

