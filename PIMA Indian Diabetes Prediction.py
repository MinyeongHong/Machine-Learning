#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Data:pima-indians-diabetes.csv
import pandas as pd

df = pd.read_csv('./dataset/pima-indians-diabetes.csv',
                 names=[ "pregnant","plasma","pressure","thickness","insulin","BMI","pedigree","age","class"])


# In[10]:


print(df.head(10))


# In[12]:


print(df.info())


# In[13]:


print(df.describe())


# In[18]:


print(df[['pregnant','class']].groupby(['pregnant']).mean().sort_values(by='pregnant',ascending=False))


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sb


# In[23]:


plt.figure(figsize=(12,12))
colormap = plt.cm.gist_heat
df.corr()


# In[24]:


sb.heatmap(df.corr(),linewidth=0.2,vmax=0.5,cmap=colormap, linecolor='white',annot=True)


# In[27]:


grid = sb.FacetGrid(df,col='class')
grid.map(plt.hist,'plasma',bins=10)
plt.show()


# In[31]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy
import tensorflow as tf

numpy.random.seed(3)
tf.random.set_seed(3)

dataset = numpy.loadtxt("./dataset/pima-indians-diabetes.csv",delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,epochs=200,batch_size=10)


# In[ ]:




