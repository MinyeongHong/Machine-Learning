#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D, Dropout,Flatten
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

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float64')/255
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float64')/255

Y_train = np_utils.to_categorical(Y_train,10)
Y_test = np_utils.to_categorical(Y_test,10)

model=Sequential()
#첫번째 컨볼루션층 (k=32, 입력 28x28x1 => 26x26x32로 변함) 
model.add(Conv2D(32,kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
#두번째 컨볼루션층 (k=64, 입력 26x26x32=> 24x24x64로 변함)
model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
#세번째 풀링층  => (12x12x64)로 변함
model.add(MaxPooling2D(pool_size=2))
#드롭아웃 25% 실시
model.add(Dropout(0.25))
#네번째 플랫튼 // 사이즈는 1차원 9216(=64x12x12)로 변함
model.add(Flatten())
#여기까지 특징 추출

#이제부터 분류 단계
model.add(Dense(128,activation='relu'))
#드롭아웃 50% 실시
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

MODEL_DIR='./model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
modelpath="./model/{epoch:02d}-{val_loss: .4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath,monitor='val_loss',verboss=1,save_best_only=True)
early_stopping_callback =EarlyStopping(monitor='val_loss', patience=10)    

history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=30,batch_size=100,verbose=0,callbacks=[early_stopping_callback, checkpointer])

print("\n Test Accuracy: %.4f" %(model.evaluate(X_test,Y_test)[1]))

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




