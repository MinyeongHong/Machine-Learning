#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D,MaxPooling1D

seed=0
numpy.random.seed(seed)
tf.random.set_seed(seed)

from keras.datasets import imdb
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=5000)

#원 핫 인코딩
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test=sequence.pad_sequences(x_test, maxlen=100)


model=Sequential()
model.add(Embedding(input_dim=5000,output_dim=100, input_length=100))
model.add(Dropout(0.5))
model.add(Conv1D(64,5,padding= 'valid',activation='relu',strides=1))
#커널 사이즈가 5 -> 양쪽으로 2만큼 사이즈 감소(=96)
#커널 개수가 64 -> 1차원 배열 64개 생성
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=100,epochs=20,validation_data=(x_test,y_test))

#테스트셋과 학습셋의 오차 설정
y_vloss=history.history['val_loss']
y_loss=history.history['loss']

#그래프로 표현
x_len=numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss,marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.',c='blue',label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:




