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

train['Age'] = train['Age'].replace('NaN', np.nan)
test['Age'] = test['Age'].replace('NaN', np.nan)

k=-1
for i in train['Age']:
	k+=1
	if pd.isna(i):
		print(train.iloc[k,:])

print(train.groupby('SibSp', as_index=False)['Age'].mean())
print(train.groupby('Parch', as_index=False)['Age'].mean())

correlation = train.groupby('SibSp', as_index=False)['Age'].mean()
print(correlation)

train = train.drop(["PassengerId","Cabin","Ticket","Name","Fare"], axis=1)
test = test.drop(["PassengerId","Cabin","Ticket","Name","Fare"], axis=1)

print(train.describe())

#Create Heatmap for data

trainMatrix = train.corr()
sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap.png')
plt.close
x=0

train[['Age']].hist(bins=10)
plt.savefig('graphs/age_hist.png')

train['Age_bins'] = pd.cut(x=train['Age'], bins=[0, 18, 25, 35, 50, 100], labels=False)
test['Age_bins'] = pd.cut(x=test['Age'], bins=[0, 18, 25, 35, 50, 100], labels=False)

train[['Age_bins']].hist(bins=5)
plt.savefig('graphs/age_bins_hist.png')

correlation = train['Age_bins'].corr(train['Survived'])
correlation2 = train['Age'].corr(train['Survived'])

print(correlation)
print(correlation2)


##print(train.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'median']).round(1))

#####deal with nan



columns_nan = ('Age','Age_bins','Embarked')

for column in columns_nan:
	if column == 'Embarked':
		strat ='most_frequent'
	else:
		strat ='mean'
	imputer = SimpleImputer(missing_values=np.nan, strategy=strat)
	imputer.fit(train[[column]])
	train[[column]]= imputer.transform(train[[column]])

	imputer = SimpleImputer(missing_values=np.nan, strategy=strat)
	imputer.fit(test[[column]])
	test[[column]]= imputer.transform(test[[column]])



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

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('Accuracy:',acc_log)