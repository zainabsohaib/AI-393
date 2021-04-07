#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# In[7]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import scipy.stats as s
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import CategoricalNB


# In[8]:



#Loading the data
df=pd.read_csv("../input/digit-recognizer/train.csv")
df


# In[ ]:


df.columns
df["label"].value_counts()


# In[ ]:


#Checking for null Values
df.isnull().sum()


# In[ ]:


#Checking dataframe for any invalid/missing data
df.info()


# In[ ]:


#Creating X & Y for Train_test_split
X = df.drop(["label"], axis=1)
y = df["label"]
X = X / 255


# In[ ]:


#spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[ ]:


y_test_pred_reg = reg.predict(X_test)
metrics.r2_score(y_test, y_test_pred_reg)


# In[ ]:


gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
bnb = BernoulliNB()
y_pred_bnb = bnb.fit(X_train, y_train).predict(X_test)
mnb = MultinomialNB()
y_pred_mnb = mnb.fit(X_train, y_train).predict(X_test)


# In[ ]:


accuracy_gnb = metrics.accuracy_score(y_test, y_pred_gnb)
accuracy_bnb = metrics.accuracy_score(y_test, y_pred_bnb)
accuracy_mnb = metrics.accuracy_score(y_test, y_pred_mnb)
print("Accuracy of GaussianNB: ",accuracy_gnb," Accuracy of BernoulliNB: ",accuracy_bnb," Accuracy of MultinomialNB: ",accuracy_mnb)


# In[ ]:


svm_clf = SVC(kernel="rbf", random_state=42, verbose=3,C=9)
svm_clf.fit(X_train, y_train)


# In[ ]:


#predicting the splited data
y_test_pred_svm = svm_clf.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, y_test_pred_svm)


# In[ ]:


test=pd.read_csv("../input/digit-recognizer/test.csv")
test=test/255
svmFinalpred=svm_clf.predict(test)


# In[ ]:


finalPred=pd.DataFrame(svmFinalpred,columns=["Label"])


# In[ ]:


finalPred['ImageId']=finalPred.index+1
finalPred = finalPred.reindex(['ImageId','Label'], axis=1)


# In[ ]:


finalPred.to_csv('./submition.csv',index=False)


# In[ ]:





# In[ ]:




