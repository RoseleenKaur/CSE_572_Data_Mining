# -*- coding: utf-8 -*-

import pandas as pd

def pipeline(df):  
    realcount=df[['Date','Time']].groupby('Date',as_index=False).count()
    ignore_dates=realcount[(realcount['Time']<270) | (realcount['Time']>300)]['Date']
    ignore_dates=ignore_dates.tolist()
    df=df[~df['Date'].isin(ignore_dates)]
    
    realcount=df[['Date','Time']].groupby('Date',as_index=False).count()
    dates_with_less_data=realcount[(realcount['Time']<288)]['Date']
    for date in dates_with_less_data.tolist():
        temp=df[df['Date']==date]
        count=temp.shape[0]
        s=temp.sample(n=(288-count))
        df=df.append(s)
        
    realcount=df[['Date','Time']].groupby('Date',as_index=False).count()
    dates_with_more_data=realcount[(realcount['Time']>288)]['Date']
    for date in dates_with_more_data.tolist():
        temp=df[df['Date']==date]
        count=temp.shape[0]
        s=temp.sample(n=(count-288))
        df.drop(s.index.tolist(),0,inplace=True)
    return df


def calculate_metrics(auto_data,manual_data,filter_DayTime,glucosefilter,columnname,daytime_name):
    data=auto_data[filter_DayTime & glucosefilter]
    result=data[['Date',"Sensor Glucose (mg/dL)"]].groupby('Date').count()
    result['mean_perday']=result["Sensor Glucose (mg/dL)"]/288
    answer_automode=result['mean_perday'].sum()/auto_no_Of_dates
    
    data=manual_data[filter_DayTime & glucosefilter]
    result=data[['Date',"Sensor Glucose (mg/dL)"]].groupby('Date').count()
    result['mean_perday']=result["Sensor Glucose (mg/dL)"]/288
    answer_manualmode=result['mean_perday'].sum()/manual_no_Of_dates
    
    ser=pd.Series(data=[answer_manualmode,answer_automode],index=['Manual Mode','Auto Mode'])
    answer_df[columnname+" "+daytime_name]=ser
   


CGMData=pd.read_csv("CGMData.csv")



CGMData=CGMData[['Index','Date','Time',"Sensor Glucose (mg/dL)"]]
CGMData['DateTime']= pd.to_datetime(CGMData['Date']+' '+CGMData['Time'], format='%m/%d/%Y %H:%M:%S')



CGMData["Sensor Glucose (mg/dL)"]=CGMData["Sensor Glucose (mg/dL)"].interpolate(method ='linear', limit_direction ='both') 


InsulinData=pd.read_csv("InsulinData.csv")


InsulinData=InsulinData[['Index','Date','Time',"Alarm"]]
InsulinData['DateTime']= pd.to_datetime(InsulinData['Date']+' '+InsulinData['Time'], format='%m/%d/%Y %H:%M:%S')


automode=InsulinData[InsulinData["Alarm"]=='AUTO MODE ACTIVE PLGM OFF']


Insulin_auto_mode=automode['DateTime'].min()


CGM_auto_mode=CGMData[CGMData['DateTime']>=Insulin_auto_mode]['DateTime'].min()


Auto_CGMData=CGMData[CGMData['DateTime']>=CGM_auto_mode]


Manual_CGMData=CGMData[CGMData['DateTime']<CGM_auto_mode]


answer_df=pd.DataFrame()


filter_manualmode=CGMData['DateTime']<CGM_auto_mode
filter_automode  =CGMData['DateTime']>=CGM_auto_mode
filter_DayTime   =(CGMData['DateTime'].dt.hour>=6)
filter_midnight  =(CGMData['DateTime'].dt.hour<6)
filter_wholeday  =(CGMData['DateTime'].dt.hour>=0)
filter_hyperglycemia =CGMData["Sensor Glucose (mg/dL)"]>180
filter_hyperglycemia_critical =CGMData["Sensor Glucose (mg/dL)"]>250
filter_range1 =(CGMData["Sensor Glucose (mg/dL)"]>=70) & (CGMData["Sensor Glucose (mg/dL)"]<=180)
filter_range_secondary =(CGMData["Sensor Glucose (mg/dL)"]>=70) & (CGMData["Sensor Glucose (mg/dL)"]<=150)
filter_hypoglycemia_level_1 =CGMData["Sensor Glucose (mg/dL)"]<70
filter_hyperglycemia_level_2 =CGMData["Sensor Glucose (mg/dL)"]<54


Auto_CGMData=pipeline(Auto_CGMData)
Manual_CGMData=pipeline(Manual_CGMData)

auto_data1=Auto_CGMData[['Date',"Sensor Glucose (mg/dL)",'Time']].groupby('Date').count()
auto_no_Of_dates=auto_data1.shape[0]


manual_data=Manual_CGMData[['Date',"Sensor Glucose (mg/dL)"]].groupby('Date').count()
manual_no_Of_dates=manual_data.shape[0]


calculate_metrics(Auto_CGMData,Manual_CGMData,filter_DayTime,filter_hyperglycemia,'Percentage time in hyperglycemia (CGM > 180 mg/dL)','(DayTime)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_DayTime,filter_hyperglycemia_critical,'percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','(DayTime)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_DayTime,filter_range1,'percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','(DayTime)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_DayTime,filter_range_secondary,'percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','(DayTime)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_DayTime,filter_hypoglycemia_level_1,'percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','(DayTime)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_DayTime,filter_hyperglycemia_level_2,'percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','(DayTime)')

calculate_metrics(Auto_CGMData,Manual_CGMData,filter_midnight,filter_hyperglycemia,'Percentage time in hyperglycemia (CGM > 180 mg/dL)','(Overnight)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_midnight,filter_hyperglycemia_critical,'percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','(Overnight)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_midnight,filter_range1,'percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','(Overnight)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_midnight,filter_range_secondary,'percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','(Overnight)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_midnight,filter_hypoglycemia_level_1,'percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','(Overnight)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_midnight,filter_hyperglycemia_level_2,'percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','(Overnight)')

calculate_metrics(Auto_CGMData,Manual_CGMData,filter_wholeday,filter_hyperglycemia,'Percentage time in hyperglycemia (CGM > 180 mg/dL)','(WholeDay)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_wholeday,filter_hyperglycemia_critical,'percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','(WholeDay)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_wholeday,filter_range1,'percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','(WholeDay)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_wholeday,filter_range_secondary,'percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','(WholeDay)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_wholeday,filter_hypoglycemia_level_1,'percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','(WholeDay)')
calculate_metrics(Auto_CGMData,Manual_CGMData,filter_wholeday,filter_hyperglycemia_level_2,'percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','(WholeDay)')


answer_df.to_csv('Kaur_Results.csv') 


