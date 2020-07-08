import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
IDtest = test["PassengerId"]

train = train.drop(["PassengerId","Cabin","Ticket","Name","Fare"], axis=1)
test = test.drop(["PassengerId","Cabin","Ticket","Name","Fare"], axis=1)

print(train.describe())

#Create Heatmap for data

trainMatrix = train.corr()
sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap.png')

##print(train.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'median']).round(1))

#####deal with nan

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(train[['Age']])
train[['Age']]= imputer.transform(train[['Age']])

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(test[['Age']])
test[['Age']]= imputer.transform(test[['Age']])

imputers = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputers.fit(train[['Embarked']])
train[['Embarked']]= imputers.transform(train[['Embarked']])

imputers = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputers.fit(test[['Embarked']])
test[['Embarked']]= imputers.transform(test[['Embarked']])

###check for nan

#print(pd.isnull(train).sum() > 0)
#print(pd.isnull(test).sum() > 0)

#transform categorial to numerical 

label_encoder = preprocessing.LabelEncoder() 

train['Sex']= label_encoder.fit_transform(train['Sex']) 
train['Embarked']= label_encoder.fit_transform(train['Embarked'])

test['Sex']= label_encoder.fit_transform(test['Sex']) 
test['Embarked']= label_encoder.fit_transform(test['Embarked'])

#define training and test sets

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.copy()

#fit data

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('Accuracy:',acc_log)