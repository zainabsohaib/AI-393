import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("train.csv")
print(dataset.head(5))
print("Total number of passengers: ",str(len(dataset.index)))
sb.countplot(x="Survived",hue="Sex",data=dataset)
plt.show()

dataset["Fare"].plot.hist()
plt.show()
sb.countplot(x="Parch",data=dataset)
plt.show()

print(dataset.isnull())
print(dataset.isnull().sum())
sb.heatmap(dataset.isnull(),yticklabels=False,cbar=False)
plt.show()
dataset.drop("Cabin",axis=1,inplace=True)
dataset.dropna(inplace=True)
sb.heatmap(dataset.isnull(),yticklabels=False,cbar=False)
plt.show()

print(pd.get_dummies(dataset["Sex"]))
sex = pd.get_dummies(dataset["Sex"],drop_first=True)
embark = pd.get_dummies(dataset["Embarked"],drop_first=True)
pcl = pd.get_dummies(dataset["Pclass"],drop_first=True)
dataset = pd.concat([dataset,sex,embark,pcl],axis=1)

print(dataset.head(5))
dataset.drop(["Sex","Pclass","Embarked","PassengerId","Name","Ticket"],axis=1,inplace=True)

print(dataset.head(5))
y = dataset["Survived"]
x = dataset.drop("Survived",axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=1)
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)

print(classification_report(Y_test,predictions))
print(accuracy_score(Y_test,predictions))
