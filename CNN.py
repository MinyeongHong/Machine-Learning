#!/usr/bin/env python
# coding: utf-8

# In[48]:


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
import numpy
import sys
import os
import tensorflow as tf

seed=3
numpy.random.seed(seed)
tf.random.set_seed(seed)

(X_train,Y_train),(X_test, Y_test) = mnist.load_data()

'''plt.imshow(X_train[0],cmap='Greys')
plt.show()

for x in X_train[0]:
    for i in x:
        sys.stdout.write('%d\t' %i)
    sys.stdout.write('\n')'''
    
#print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0],784)
#print(X_train.shape)
X_train = X_train.astype('float64')/255
X_test = X_test.reshape(X_test.shape[0],784).astype('float64')/255
#print(Y_train[0])
Y_train = np_utils.to_categorical(Y_train,10)
Y_test = np_utils.to_categorical(Y_test,10)

model=Sequential()
model.add(Dense(512,input_dim=784, activation='relu'))
model.add(Dense(10,activation='softmax'))
#784개의 입력을 512개로 노드로 줄인 뒤 다시 10개로 줄이고 그중 가장 큰 값 출력

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

MODEL_DIR='./model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
modelpath="./model/{epoch:02d}-{val_loss: .4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath,monitor='val_loss',verboss=1,save_best_only=True)
early_stopping_callback =EarlyStopping(monitor='val_loss', patience=10)    

history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=30,batch_size=100,verbose=0,callbacks=[early_stopping_callback, checkpointer])

print("\n Test Accuracy: %.4f" %(model.evaluate(X_test,Y_test)[1]))

y_vloss=history.history['val_loss']
y_loss=history.history['loss']

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





# In[ ]:




