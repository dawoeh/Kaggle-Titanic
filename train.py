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

titels = ('Mlle.', 'Dr.', 'Master.', 'Col.', 'Rev.', 'Sir.', 'Lady.', 'Major.', 'Mme.', 'Capt.')

k=-1
for i in train['Age']:
	k+=1
	string = (train.iloc[k,3])
	train.at[k, 'Name'] = string.split(',', 1)[0]
	train.at[k, 'Titles'] = 0
	for i in string.split(' '):
		if i in titels:
			train.at[k, 'Titles'] = 1	
	if pd.isna(i):
		if train.iloc[k,6] < 2:
			if train.iloc[k,7] > 2:
				train.at[k, 'Age'] = train.loc[train['Parch'] > 2, 'Age'].mean()
			else:
				train.at[k, 'Age'] = train['Age'].mean()
		if train.iloc[k,6] == 2:
			train.at[k, 'Age'] = train.loc[train['SibSp'] == 2, 'Age'].mean()
		if train.iloc[k,6] > 2:
			train.at[k, 'Age'] = train.loc[train['SibSp'] > 2, 'Age'].mean()

k=-1
for i in test['Age']:
	k+=1
	string = (test.iloc[k,2])
	test.at[k, 'Name'] = string.split(',', 1)[0]
	test.at[k, 'Titles'] = 0
	for i in string.split(' '):
		if i in titels:
			test.at[k, 'Titles'] = 1	
	if pd.isna(i):
		if test.iloc[k,5] < 2:
			if test.iloc[k,6] > 2:
				test.at[k, 'Age'] = train.loc[train['Parch'] > 2, 'Age'].mean()
			else:
				test.at[k, 'Age'] = train['Age'].mean()
		if test.iloc[k,5] == 2:
			test.at[k, 'Age'] = train.loc[train['SibSp'] == 2, 'Age'].mean()
		if test.iloc[k,5] > 2:
			test.at[k, 'Age'] = train.loc[train['SibSp'] > 2, 'Age'].mean()

print(train.groupby('SibSp', as_index=False)['Age'].mean())
print(train.groupby('Parch', as_index=False)['Age'].mean())


train['Age_bins'] = pd.cut(x=train['Age'], bins=[0, 18, 24, 30, 50, 80], labels=False)
test['Age_bins'] = pd.cut(x=test['Age'], bins=[0, 18, 24, 30, 50, 80], labels=False)

##print(correlation)

print(train.describe())

#Create Graphs and Heatmap for data

train[['Age']].hist(bins=10)
plt.savefig('graphs/age_hist.png')


train[['Age_bins']].hist(bins=5)
plt.savefig('graphs/age_bins_hist.png')

train = train.drop(["PassengerId","Cabin","Ticket","Name","Fare"], axis=1)
test = test.drop(["PassengerId","Cabin","Ticket","Name","Fare"], axis=1)

plt.close

trainMatrix = train.corr()
sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap.png')
plt.close
x=0

#####deal with nan

#print(pd.isnull(train).sum() > 0)
#print(pd.isnull(test).sum() > 0)

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