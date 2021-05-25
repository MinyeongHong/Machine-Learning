#!/usr/bin/env python
# coding: utf-8

# In[16]:


from keras.models import Sequential,load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold

import pandas as pd
import numpy
import tensorflow as tf

#seed 값 설정
seed=3
numpy.random.seed(3)
tf.random.set_seed(3)

#데이터 입력
df=pd.read_csv('./dataset/sonar.csv',header=None)

#print(df.info())
#print(df.describe())

dataset=df.values
X=dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

e=LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=seed)

#교차 검증 방법
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy=[]
for train,test in skf.split(Y,Y):
    model=Sequential()
    model.add(Dense(24,input_dim=60,activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X[train],Y[train],epochs=200,batch_size=5)
    k_accuracy='%.4f'%(model.evaluate(X[test],Y[test])[1])
    accuracy.append(k_accuracy)
    
print("\n %.f fold Accuracy:"%n_fold,accuracy)

'''정확도가 높은 지점의 데이터를 저장
model.save('my_model.h5')
#삭제 후 다시 불러오기가 가능하다
del model
model2 = load_model('my_model.h5')
print("\n Accuracy: %.4f"%(model2.evaluate(X_test,Y_test)[1]))'''


# In[ ]:




