#MULTINOMIAL NB ON 5*5 WEIGHTED
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
train = pd.read_csv("train.csv")  # reading data from dataset ,
#dataset from kaggle digit recognizer competition
X = train.drop('label',axis=1)
Y = train['label']
# the filter for convolution 5*5 (unweighted)

filter = np.array([[1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1]])

X = X.to_numpy()  #conversion of datafrom to numpy array
print(X.shape)

#that new array which is reduced
size = np.empty((0,576), int)

#creating a function which can perform convolution 
def convolution(image, filter):
  fX, fY = filter.shape 
# for getting filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2)
    #mentioned new dimensions of image which has been reduced
newImage = np.zeros((nn,nn))
#take new and empty 2D image
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

subset = 500 
#subset size for dry runs change to 42000 to run on whole data
#convolving all images in below code
for img in X[0:subset,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolution(img2D,filter)
 
  nImg1D = np.reshape(nImg, (-1,576))
 
  sX= np.append(sX, nImg1D, axis=0) #size

Y = Y.to_numpy()
sY = Y[0:subset]
# print(sY)
print(sY.shape)
print(sX.shape)
#test and train your data here 
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
#work for MULTINOMIAL NAIVE BAYES
check = MultinomialNB()
check.fit(sXTrain, yTrain)
print(check.class_count_)
print(check.score(sXTest, yTest))

#MULTINOMIAL NB 5*5 WEIGHTED
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
train = pd.read_csv("train.csv")  # reading data from dataset ,
#dataset from kaggle digit recognizer competition
X = train.drop('label',axis=1)
Y = train['label']
#Create Filter for convolution weighted 5*5
filter = np.array([[1,1,1,1,1],
[1,2,2,2,1],
[1,2,3,2,1],
[1,2,2,2,1],
[1,1,1,1,1]])


X = X.to_numpy()  #conversion of datafrom to numpy array
print(X.shape)

#that new array which is reduced
size = np.empty((0,576), int)

#creating a function which can perform convolution 
def convolution(image, filter):
  fX, fY = filter.shape 
# for getting filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2)
    #mentioned new dimensions of image which has been reduced
newImage = np.zeros((nn,nn))
#take new and empty 2D image
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

subset = 500 
#subset size for dry runs change to 42000 to run on whole data
#convolving all images in below code
for img in X[0:subset,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolution(img2D,filter)
 
  nImg1D = np.reshape(nImg, (-1,576))
 
  sX= np.append(sX, nImg1D, axis=0) #size

Y = Y.to_numpy()
sY = Y[0:subset]
# print(sY)
print(sY.shape)
print(sX.shape)
#test and train your data here 
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
#work for MULTINOMIAL NAIVE BAYES
check = MultinomialNB()
check.fit(sXTrain, yTrain)
print(check.class_count_)
print(check.score(sXTest, yTest))

#multinomialNB 7 by 7 un weighted filter

import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
train = pd.read_csv("train.csv")  # reading data from dataset ,
#dataset from kaggle digit recognizer competition
X = train.drop('label',axis=1)
Y = train['label']
filter = np.array([[1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1]])

#above is the filter for convolution 7 by 7 (unweighted)



X = X.to_numpy()  #conversion of datafrom to numpy array
print(X.shape)

#that new array which is reduced
size = np.empty((0,484), int)

#creating a function which can perform convolution 
def convolution(image, filter):
  fX, fY = filter.shape 
# for getting filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2)
    #mentioned new dimensions of image which has been reduced
newImage = np.zeros((nn,nn))
#take new and empty 2D image
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

subset = 500 
#subset size for dry runs change to 42000 to run on whole data
#convolving all images in below code
for img in X[0:subset,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolution(img2D,filter)
 
  nImg1D = np.reshape(nImg, (-1,484))
 
  sX= np.append(sX, nImg1D, axis=0) #size

Y = Y.to_numpy()
sY = Y[0:subset]
# print(sY)
print(sY.shape)
print(sX.shape)
#test and train your data here 
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
#work for MULTINOMIAL NAIVE BAYES
check = MultinomialNB()
check.fit(sXTrain, yTrain)
print(check.class_count_)
print(check.score(sXTest, yTest))

#multinomialNB 7 by 7 weighted filter

import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
train = pd.read_csv("train.csv")  # reading data from dataset ,
#dataset from kaggle digit recognizer competition
X = train.drop('label',axis=1)
Y = train['label']
filter = np.array([[1,1,1,1,1,1,1],
          [1,2,2,2,2,2,1],
          [1,2,3,3,3,2,1],
          [1,2,3,4,3,2,1],
          [1,2,3,3,3,2,1],
          [1,2,2,2,2,2,1],
          [1,1,1,1,1,1,1]])

#convolution 7 by 7 (weighted)



X = X.to_numpy()  #conversion of datafrom to numpy array
print(X.shape)

#that new array which is reduced
size = np.empty((0,484), int)

#creating a function which can perform convolution 
def convolution(image, filter):
  fX, fY = filter.shape 
# for getting filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2)
    #mentioned new dimensions of image which has been reduced
newImage = np.zeros((nn,nn))
#take new and empty 2D image
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

subset = 500 
#subset size for dry runs change to 42000 to run on whole data
#convolving all images in below code
for img in X[0:subset,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolution(img2D,filter)
 
  nImg1D = np.reshape(nImg, (-1,484))
 
  sX= np.append(sX, nImg1D, axis=0) #size

Y = Y.to_numpy()
sY = Y[0:subset]
# print(sY)
print(sY.shape)
print(sX.shape)
#test and train your data here 
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
#work for MULTINOMIAL NAIVE BAYES
check = MultinomialNB()
check.fit(sXTrain, yTrain)
print(check.class_count_)
print(check.score(sXTest, yTest))

#9 by 9 unweighted

import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
train = pd.read_csv("train.csv")  # reading data from dataset ,
#dataset from kaggle digit recognizer competition
X = train.drop('label',axis=1)
Y = train['label']
filter = np.array([[1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
	        [1,1,1,1,1,1,1,1,1],
	        [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1]])

#above is the filter for convolution 9*9 (unweighted)




X = X.to_numpy()  #conversion of datafrom to numpy array
print(X.shape)

#that new array which is reduced
size = np.empty((0,400), int)

#creating a function which can perform convolution 
def convolution(image, filter):
  fX, fY = filter.shape 
# for getting filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2)
    #mentioned new dimensions of image which has been reduced
newImage = np.zeros((nn,nn))
#take new and empty 2D image
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

subset = 500 
#subset size for dry runs change to 42000 to run on whole data
#convolving all images in below code
for img in X[0:subset,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolution(img2D,filter)
 
  nImg1D = np.reshape(nImg, (-1,400))
 
  sX= np.append(sX, nImg1D, axis=0) #size

Y = Y.to_numpy()
sY = Y[0:subset]
# print(sY)
print(sY.shape)
print(sX.shape)
#test and train your data here 
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
#work for MULTINOMIAL NAIVE BAYES
check = MultinomialNB()
check.fit(sXTrain, yTrain)
print(check.class_count_)
print(check.score(sXTest, yTest))

#9 by 9 weighted
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
train = pd.read_csv("train.csv")  # reading data from dataset ,
#dataset from kaggle digit recognizer competition
X = train.drop('label',axis=1)
Y = train['label']
filter = np.array([[1,1,1,1,1,1,1,1,1],
          [1,2,2,2,2,2,2,2,1],
          [1,2,3,3,3,3,3,2,1],
          [1,2,3,4,4,4,3,2,1],
          [1,2,3,4,5,4,3,2,1],
          [1,2,3,4,4,4,3,2,1],
	        [1,2,3,3,3,3,3,2,1],
	        [1,2,2,2,2,2,2,2,1],
          [1,1,1,1,1,1,1,1,1]])




X = X.to_numpy()  #conversion of datafrom to numpy array
print(X.shape)

#that new array which is reduced
size = np.empty((0,400), int)

#creating a function which can perform convolution 
def convolution(image, filter):
  fX, fY = filter.shape 
# for getting filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2)
    #mentioned new dimensions of image which has been reduced
newImage = np.zeros((nn,nn))
#take new and empty 2D image
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

subset = 500 
#subset size for dry runs change to 42000 to run on whole data
#convolving all images in below code
for img in X[0:subset,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolution(img2D,filter)
 
  nImg1D = np.reshape(nImg, (-1,400))
 
  sX= np.append(sX, nImg1D, axis=0) #size

Y = Y.to_numpy()
sY = Y[0:subset]
# print(sY)
print(sY.shape)
print(sX.shape)
#test and train your data here 
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
#work for MULTINOMIAL NAIVE BAYES
check = MultinomialNB()
check.fit(sXTrain, yTrain)
print(check.class_count_)
print(check.score(sXTest, yTest))


