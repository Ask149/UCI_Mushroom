# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


SEED = 8888
np.random.seed(SEED)
columns = ["edible", "cap-shape", "cap-surface", "cap-color", "bruises?",
            "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
            "stalk-shape", "stalk-root", "stalk-surface-above-ring",
            "stalk-surface-below-ring", "stalk-color-above-ring",
            "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
            "ring-type", "spore-print-color", "population", "habitat"
            ]
dataset = pd.read_csv('mycsv.csv',names=columns)


# In[3]:


#for value in columns:
#    print (value,":", sum(dataset[value] == '?'))
df_rev = dataset
for value in columns:
    df_rev[value].replace(['?'], [df_rev.describe(include='all')[value][2]],inplace=True)


# In[4]:


df_rev = df_rev.apply(LabelEncoder().fit_transform)


# In[5]:


features = df_rev.values[:,1:]
target = df_rev.values[:,0]
#print(features)
#print(target)


# In[6]:


columns1 = ["cap-shape", "cap-surface", "cap-color", "bruises?",
            "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
            "stalk-shape", "stalk-root", "stalk-surface-above-ring",
            "stalk-surface-below-ring", "stalk-color-above-ring",
            "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
            "ring-type", "spore-print-color", "population", "habitat"
            ]
scaled_features = {}
for each in columns1:
    mean , std = df_rev[each].mean(), df_rev[each].std()
    scaled_features[each] = [mean,std]
    df_rev.loc[:, each] = (df_rev[each]-mean)/std


# In[7]:


features_train, features_test, target_train, target_test = train_test_split(features,target, test_size = 0.66,random_state = 10)


# In[8]:


clf = SVC()
clf.fit(features_train,target_train)
target_pred = clf.predict(features_test)


# In[9]:


accuracy_score(target_test, target_pred, normalize = True)