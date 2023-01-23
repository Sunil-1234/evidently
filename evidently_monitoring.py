#!/usr/bin/env python
# coding: utf-8
---
title: VMONITOR
description: Data Drift Detection in classification task
show-code: False
params:
    new_samples:
        input: slider
        value: 25
        label: New samples count
        min: 10
        max: 75
    verbose:
        input: checkbox
        value: False
        label: Verbose 
---
# In[1]:


import warnings
warnings.filterwarnings('ignore')
#import streamlit as st
from IPython.display import display


# In[2]:


from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab,NumTargetDriftTab,ClassificationPerformanceTab
from evidently import ColumnMapping
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, CatTargetDriftProfileSection


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv('kipu_biasing_processed.csv')


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


user_data_drift_test = df[['gender','ethnicity'	,'addressstate',  'age']]
user_data_drift_dashboard = Dashboard(tabs=[DataDriftTab(verbose_level=1)])
user_data_drift_dashboard.calculate(user_data_drift_test[:3000],  user_data_drift_test[3000:], column_mapping=None)
user_data_drift_dashboard.show(mode='inline')
#user_data_drift_dashboard.save('stroke_data_drift_dashboard.html')


# In[9]:


target_column_mapping = ColumnMapping()
target_column_mapping.target = 'target'
target_column_mapping.numerical_features = []
ref_data_sample = df[:3000].sample(1000, random_state=0)
prod_data_sample = df[3000:].sample(1000, random_state=0)
rating_target_drift_dashboard = Dashboard(tabs=[CatTargetDriftTab(verbose_level=1)])
rating_target_drift_dashboard.calculate(ref_data_sample, prod_data_sample, column_mapping=target_column_mapping)
rating_target_drift_dashboard.show(mode='inline')


# In[10]:


ele = -1
def sunil(x, dic):
    global ele
    if x in dic.keys():
        return dic[x]
    else:
        ele += 1
        dic[x] = ele
        return dic[x]
    
def somil(df):
    mapped = dict()
    df.dropna(inplace=True)
    for column in df.columns:
        if df[column].dtype == 'O':
            global ele
            ele = 0
            dic = dict()
            df[column] = df[column].apply(lambda x: sunil(x, dic))
            dic = {v:k for k, v in dic.items()}
            mapped[column] = dic
    return mapped


# In[11]:


import pandas as pd
df=pd.read_csv('kipu_biasing_processed.csv')
dic=somil(df)
X=df.drop(['patientmasterkey','target','cpt_code','level_of_care','diagcodename'],axis=1)
y=df[['target']]
#dic=somil(df)
target = 'target'
prediction = 'prediction'
numerical_features =X.select_dtypes(include=["int",'float']).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# In[12]:


def reverse_encode(dic, df):
    for key in dic.keys():
        df[key] = df[key].apply(lambda x: dic[key][x])


# In[13]:


from sklearn import  ensemble
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard import Dashboard

from evidently.dashboard.tabs import DataDriftTab, NumTargetDriftTab, RegressionPerformanceTab
reference = df.loc[:15000]
current=df.loc[15000:]


regressor = ensemble.RandomForestClassifier(random_state = 0, n_estimators = 50)
regressor.fit(reference[numerical_features + categorical_features], reference[target])
ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
current_prediction = regressor.predict(current[numerical_features + categorical_features])
current_predprob=regressor.predict_proba(current[numerical_features + categorical_features])
reference['prediction'] = ref_prediction
current['prediction'] = current_prediction


pred_prob=[]
for i in range(0,current.shape[0]):
    #print(i)
    pred_prob.insert(i,current_predprob[i][1])
current['pred_prob'] =pred_prob




# In[14]:


column_mapping = ColumnMapping()
reverse_encode(dic,reference)
reverse_encode(dic,current)
# reference=reference.astype('str')
# current=current.astype('str')
column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
#column_mapping.categorical_features = categorical_features

regression_perfomance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab(verbose_level=1)])
regression_perfomance_dashboard.calculate(reference.astype('str'), current.astype('str'), column_mapping=column_mapping)

regression_perfomance_dashboard.show(mode='inline')


# In[15]:


#current.loc[current['age'] < 20, 'stroke_risk'] = [np.random.uniform(0.0, 0.2) for i in range(196)]


# In[16]:


import seaborn as sns
data=current[['age','target']][300:400]
#print(data.shape)
plt.rcParams["figure.figsize"] = (20,3)
#plt.xticks(data['age'][::20])  
sns.lineplot(x='age',y='target',data=data)
plt.show()


# In[ ]:





# In[ ]:




