#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

seed=0
numpy.random.seed(seed)
tf.random.set_seed(seed)

from keras.datasets import reuters

(X_train,Y_train),(X_test,Y_test) = reuters.load_data(num_words=1000,test_split=0.2)

#원 핫 인코딩
x_train = sequence.pad_sequences(X_train, maxlen=100)
x_test=sequence.pad_sequences(X_test, maxlen=100)

y_train = np_utils.to_categorical(Y_train)
y_test= np_utils.to_categorical(Y_test) 

model=Sequential()
model.add(Embedding(input_dim=1000,output_dim=100, input_length=100))
model.add(LSTM(100,activation='tanh'))
model.add(Dense(46,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




