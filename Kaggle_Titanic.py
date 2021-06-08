#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import sklearn

import warnings
warnings.filterwarnings('ignore') 

df_train=pd.read_csv("/kaggle/input/titanic/train.csv")
df_test= pd.read_csv("/kaggle/input/titanic/test.csv")

#df_train.info()

'''
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
df_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=True).mean().plot.bar()
df_train[['Parch', 'Survived']].groupby(['Parch'], as_index=True).mean().plot.bar()
df_train[['Ticket', 'Survived']].groupby(['Ticket'], as_index=True).mean().plot.bar()
df_train[['Fare', 'Survived']].groupby(['Fare'], as_index=True).mean().plot.bar()
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().plot.bar()

df_train.hist(figsize=(8,8))
'''


#-----------데이터 가공하기------------
#Sex Data 가공 : 성별 전처리.문자열을 숫자로 
Sex_replace = {'male':0,'female':1}
df_train=df_train.replace({'Sex':Sex_replace})
df_test=df_test.replace({'Sex':Sex_replace})


#Name Data 가공 : 이름에서 호칭 따오기
df_train['Title']= df_train.Name.str.extract('([A-Za-z]+)\.')
df_test['Title']= df_test.Name.str.extract('([A-Za-z]+)\.') 

pd.crosstab(df_train['Title'], df_train['Sex']).T.style.background_gradient(cmap='summer_r') 

df_train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby('Title').mean()

#SibSp & Parch Data 가공 : 두개의 데이터 합치기
df_train['NumOfFam'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['NumOfFam'] = df_test['SibSp'] + df_test['Parch'] + 1 

#Age Data 가공 
sns.factorplot(y="Age",x="Pclass",data=df_train,size=2,aspect=1.5)
g = sns.factorplot(y="Age",x="Title",data=df_train,kind="box")
g = sns.factorplot(y="Age",x="Title",hue="Pclass", data=df_train,kind="box")
# => Pclass 등급이 낮아질 수록(1>2>3) 연령대가 어려짐, 타이틀에 따라 평균 연령이 다름
#클래스와 타이틀을 고려한 평균 연령 계산
df_train.groupby(['Title','Pclass'])['Age'].mean()
df_test.groupby(['Title','Pclass'])['Age'].mean()
# Null 데이터의 인덱스 찾기
idx_nullage = list(df_train["Age"][df_train["Age"].isnull()].index)
idx_nullage_test = list(df_test["Age"][df_test["Age"].isnull()].index)
#평균 연령으로 NULL값을 채워줌

df_train.groupby(['Title','Pclass'])['Age'].mean()
df_test.groupby(['Title','Pclass'])['Age'].mean()

for i in idx_nullage :
    if df_train.loc[i,'Pclass']==1:
        df_train.loc[(df_train.Title=='Mr'),'Age'] = 42
        df_train.loc[(df_train.Title=='Mrs'),'Age'] = 41
        df_train.loc[(df_train.Title=='Master'),'Age'] = 5
        df_train.loc[(df_train.Title=='Miss'),'Age'] = 30
        df_train.loc[(df_train.Title=='Other'),'Age'] = 51
        
    elif df_train.loc[i,'Pclass']==2:
        df_train.loc[(df_train.Title=='Mr'),'Age'] = 33
        df_train.loc[(df_train.Title=='Mrs'),'Age'] = 34
        df_train.loc[(df_train.Title=='Master'),'Age'] = 2
        df_train.loc[(df_train.Title=='Miss'),'Age'] = 23
        df_train.loc[(df_train.Title=='Other'),'Age'] = 43
        
    else:
        df_train.loc[(df_train.Title=='Mr'),'Age'] =29
        df_train.loc[(df_train.Title=='Mrs'),'Age'] = 34
        df_train.loc[(df_train.Title=='Master'),'Age'] = 5
        df_train.loc[(df_train.Title=='Miss'),'Age'] = 16
 
for i in idx_nullage_test :
    if df_test.loc[i,'Pclass']==1:
        df_test.loc[(df_test.Title=='Mr'),'Age'] = 41
        df_test.loc[(df_test.Title=='Mrs'),'Age'] = 46
        df_test.loc[(df_test.Title=='Master'),'Age'] = 10
        df_test.loc[(df_test.Title=='Miss'),'Age'] = 31
        df_test.loc[(df_test.Title=='Other'),'Age'] = 50
        
    elif df_test.loc[i,'Pclass']==2:
        df_test.loc[(df_test.Title=='Mr'),'Age'] = 32
        df_test.loc[(df_test.Title=='Mrs'),'Age'] = 33
        df_test.loc[(df_test.Title=='Master'),'Age'] = 5
        df_test.loc[(df_test.Title=='Miss'),'Age'] = 17
        df_test.loc[(df_test.Title=='Other'),'Age'] = 36
        
    else:
        df_test.loc[(df_test.Title=='Mr'),'Age'] = 27
        df_test.loc[(df_test.Title=='Mrs'),'Age'] = 30
        df_test.loc[(df_test.Title=='Master'),'Age'] = 7
        df_test.loc[(df_test.Title=='Miss'),'Age'] = 20

        
#String to Numerical
df_train['Title'] = df_train['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_test['Title'] = df_test['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})


#Fare Data 가공 : Null값 Pclass로 채우기
df_test["Fare"].fillna(df_test.groupby('Pclass')['Fare'].transform('median'), inplace=True)

#Embarked Data 가공 : 문자열 숫자로  & Null값에는 가장 다수의 값을 대입
Embarked_replace={'S':3,'C':2,'Q':1}
df_train = df_train.replace({'Embarked':Embarked_replace})
df_test = df_test.replace({'Embarked':Embarked_replace})

df_train['Embarked']=df_train['Embarked'].fillna(3)
df_test['Embarked']=df_test['Embarked'].fillna(3)



#One-hot encoding. 모델의 성능을 높이기 위한 작업
df_train = pd.get_dummies(df_train, columns=['Title'], prefix='Title')
df_test = pd.get_dummies(df_test, columns=['Title'], prefix='Title')
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')


#사용하지 않을 데이터 삭제하기
df_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
df_test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)

print(df_train.head())

#----------모델 제작--------------
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn import metrics
from sklearn.model_selection import train_test_split


seed=0
np.random.seed(seed)
tf.random.set_seed(seed)

X_train= df_train.drop('Survived', axis=1).values
Label = df_train['Survived'].values
X_test = df_test.values

x_tr, x_te, y_tr, y_te = train_test_split(X_train, Label, test_size=0.2, random_state=seed)

#모델1 : 시퀀셜
model=Sequential()
model.add(Dense(3,activation='relu',input_dim=13))
model.add(Dense(2,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(x_tr,y_tr,validation_split=0.2,epochs=1000,batch_size=20)

prediction_sample = model.predict(x_te).argmax(axis=1)
print('{:.2f}% 정확도'.format(100 * metrics.accuracy_score(prediction_sample, y_te)))

submission=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
prediction = model.predict(X_test)
submission['Survived'] = prediction

#도출한 생존 확률을 0과 1로 변환
series = []
for val in submission.Survived:
    if val >= 0.5:
        series.append(1)
    else:
        series.append(0)
        
submission.drop(['Survived'],axis=1,inplace=True)
submission['Survived'] = series


submission.to_csv('titanic-submission.csv',index=False)

