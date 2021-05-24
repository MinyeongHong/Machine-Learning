#!/usr/bin/env python
# coding: utf-8

# In[23]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

Data_set=np.loadtxt("./dataset/ThoraricSurgery.csv",delimiter=",")

X=Data_set[:,0:17]
Y=Data_set[:,17]

model = Sequential()
model.add(Dense(30,input_dim=17,activation='relu'))
#노드 17개로부터 30개의 중간 레이어 생성
model.add(Dense(1,activation='sigmoid'))
#출력층 생성

model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(X,Y,epochs=100,batch_size=10)

#loss function에 따라 달라지는 정확도 확인
#model.compile(loss='mean_squared_error',optimizer='adam',metrics='accuracy')
#model.fit(X,Y,epochs=100,batch_size=10)#

#optimizer에 따라 달라지는 정확도 확인
#model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics='accuracy')
#model.fit(X,Y,epochs=100,batch_size=10)


# In[ ]:





# In[ ]:




