
# coding: utf-8

# In[5]:


import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[20]:


with open('mycsv.csv', 'rt') as file:
    data = csv.reader(file,delimiter=',')
    list1= list(data)
print(list1)


# In[28]:


cnt = len(list1[0])
print(cnt)
ldata = len(list1)
print(ldata)

