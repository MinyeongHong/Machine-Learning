#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

#데이터 입력
df = pd.read_csv('./dataset/iris.csv',names = ["sepal_length", "sepal_width","petal_length","petal_width","species"])

#그래프로 확인
sns.pairplot(df,hue='species')
plt.show()


# In[5]:


#데이터 분류
dataset=df.values
X=dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]


# In[6]:


df.head()


# In[10]:


e=LabelEncoder()
e.fit(Y_obj)
Y=e.transform(Y_obj)

#one-hot encoding
Y_encoded = tf.keras.utils.to_categorical(Y)

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,Y_encoded, epochs=50, batch_size=5)
print("Accuracy: %.4f" %(model.evaluate(X,Y_encoded)[1]))


# In[ ]:




