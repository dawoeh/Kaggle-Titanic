import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn import metrics

#####import data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#####Fill empty cells with nan
print(train.columns[train.isnull().any()].tolist())
print(test.columns[test.isnull().any()].tolist())
column_nan = ('Age','Cabin','Fare')
for i in column_nan:
	train[i] = train[i].replace('NaN', np.nan)
	test[i] = test[i].replace('NaN', np.nan)


#####Show data statistics
print(train.describe())
print(test.describe())

###dataset statistics for age and fare, since data are missing
print(train.groupby('SibSp', as_index=False)['Age'].mean())
print(train.groupby('Parch', as_index=False)['Age'].mean())
print(train.groupby('Pclass', as_index=False)['Fare'].mean())

#####count titles
titles = []
for i in train['Name']:
	string = i.split(',', 1)[1]
	titles.append(string.split('.')[0])
print(Counter(titles))
titels = ('Master.','Dr.','Rev.','Major.','Mlle.','Col.','Don.','Mme.','Lady.','Sir.','Capt.', 'the Countess.', 'Jonkheer.')

######Fill empty numerical values and transform ticket into groups
k=0
for i in train['PassengerId']:
	string = (train.loc[k,'Name'])
	train.at[k, 'Name'] = string.split(',', 1)[0]
	train.at[k, 'Titles'] = 0

	for i in string.split(' '):		###extract titels from names
		if i in titels:
			train.at[k, 'Titles'] = 1

	if pd.isna(train.at[k, 'Age']):			###deal with unknown age
		if train.loc[k,'SibSp'] < 2:
			if train.loc[k,'Parch'] > 2:
				train.at[k, 'Age'] = train.loc[train['Parch'] > 2, 'Age'].mean()
			else:
				train.at[k, 'Age'] = train['Age'].mean()
		if train.loc[k,'SibSp'] == 2:
			train.at[k, 'Age'] = train.loc[train['SibSp'] == 2, 'Age'].mean()
		if train.loc[k,'SibSp'] > 2:
			train.at[k, 'Age'] = train.loc[train['SibSp'] > 2, 'Age'].mean()

	if pd.isna(train.at[k, 'Fare']):		###deal with unknown fare
		if train.at[k, 'Pclass'] == 1:
			train.at[k, 'Fare'] = train.loc[train['Pclass'] == 1, 'Fare'].mean()
		elif train.at[k, 'Pclass'] == 2:
			train.at[k, 'Fare'] = train.loc[train['Pclass'] == 2, 'Fare'].mean()
		else:
			train.at[k, 'Fare'] = train.loc[train['Pclass'] == 3, 'Fare'].mean()

	try:			####transform tickets into groups
		integer = int(train.loc[k,'Ticket'])
		if integer > 99999:
			train.at[k, 'Ticket_group'] = 0
		if integer < 100000:
			train.at[k, 'Ticket_group'] = 1
	except ValueError:
		train.at[k, 'Ticket_group'] = 2

	if pd.isna(train.at[k, 'Cabin']):
		train.at[k, 'Cabin'] = 0
	else:
		train.at[k, 'Cabin'] = 1
	k+=1

k=0
for i in test['PassengerId']:
	string = (test.loc[k,'Name'])
	test.at[k, 'Name'] = string.split(',', 1)[0]
	test.at[k, 'Titles'] = 0

	for i in string.split(' '):			###extract titels from names
		if i in titels:
			test.at[k, 'Titles'] = 1

	if pd.isna(test.at[k, 'Age']):		###deal with unknown age
		if test.loc[k,'SibSp'] < 2:
			if test.loc[k,'Parch'] > 2:
				test.at[k, 'Age'] = train.loc[train['Parch'] > 2, 'Age'].mean()
			else:
				test.at[k, 'Age'] = train['Age'].mean()
		if test.loc[k,'SibSp'] == 2:
			test.at[k, 'Age'] = train.loc[train['SibSp'] == 2, 'Age'].mean()
		if test.loc[k,'SibSp'] > 2:
			test.at[k, 'Age'] = train.loc[train['SibSp'] > 2, 'Age'].mean()

	if pd.isna(test.at[k, 'Fare']):		###deal with unknown fare
		if test.at[k, 'Pclass'] == 1:
			test.at[k, 'Fare'] = train.loc[train['Pclass'] == 1, 'Fare'].mean()
		elif test.at[k, 'Pclass'] == 2:
			test.at[k, 'Fare'] = train.loc[train['Pclass'] == 2, 'Fare'].mean()
		else:
			test.at[k, 'Fare'] = train.loc[train['Pclass'] == 3, 'Fare'].mean()

	try:		####transform tickets into groups
		integer = int(test.loc[k,'Ticket'])
		if integer > 99999:
			test.at[k, 'Ticket_group'] = 0
		if integer < 100000:
			test.at[k, 'Ticket_group'] = 1
	except ValueError:
		test.at[k, 'Ticket_group'] = 2

	if pd.isna(test.at[k, 'Cabin']):
		test.at[k, 'Cabin'] = 0
	else:
		test.at[k, 'Cabin'] = 1
	k+=1

#####create bins for age and fare
ageb=(0, 4, 16, 24, 30, 40, 80) ###bins for age
train['Age_bins'] = pd.cut(x=train['Age'], bins=ageb, labels=False)
test['Age_bins'] = pd.cut(x=test['Age'], bins=ageb, labels=False)

fareb=(-np.inf, 8.0, 16.0, 50.0, 1000) ###bins for fare
train['Fare_bins'] = pd.cut(x=train['Fare'], include_lowest=True, bins=fareb, labels=False)
test['Fare_bins'] = pd.cut(x=test['Fare'], include_lowest=True, bins=fareb, labels=False)

#####Create Graphs and Heatmap for data
train[['Age']].hist(bins=10)
plt.savefig('graphs/age_hist.png')

train[['Age_bins']].hist(bins=6)
plt.savefig('graphs/age_bins_hist.png')

train[['Fare_bins']].hist(bins=4)
plt.savefig('graphs/fare_bins_hist.png')

train = train.drop(["PassengerId","Ticket","Name","Fare","Age"], axis=1)
test = test.drop(["PassengerId","Ticket","Name","Fare","Age"], axis=1)

train['Cabin']=pd.to_numeric(train['Cabin'])
test['Cabin']=pd.to_numeric(test['Cabin'])

print(train.describe())
print(test.describe())
plt.close

trainMatrix = train.corr()

f, ax = plt.subplots(figsize=(12, 10))

sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap.png')
plt.close
x=0

#####deal with nan embarked
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(train[['Embarked']])
train[['Embarked']]= imputer.transform(train[['Embarked']])

#####transform categorial to numerical 
label_encoder = preprocessing.LabelEncoder() 

train['Sex']= label_encoder.fit_transform(train['Sex']) 
train['Embarked']= label_encoder.fit_transform(train['Embarked'])

test['Sex']= label_encoder.fit_transform(test['Sex']) 
test['Embarked']= label_encoder.fit_transform(test['Embarked'])

#####define training and test sets
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.copy()
Y_test = pd.read_csv("gender_submission.csv")
Y_test = Y_test.drop("PassengerId", axis=1)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=666)

#####fit data LogReg
logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
print('Accuracy Linear Regression:',acc_log)

#####fit data Random Forest
clf_simple=RandomForestClassifier(n_estimators= 100, random_state=666)
clf_simple.fit(x_train,y_train)
Y_pred=clf_simple.predict(x_test)
print("Accuracy Simple Random Forest:",metrics.accuracy_score(y_test, Y_pred))

#####Optimize Hyperparameters
# clf=RandomForestClassifier(random_state=666)
# param_grid = { 
#     'n_estimators': [100, 200],
#     #'min_samples_split': [2, 5, 10, 15, 100],
# 	#'min_samples_leaf': [1, 2, 5],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [3,4,5,6],
#     'criterion' :['gini', 'entropy'] 
# }
# CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
# CV_clf.fit(x_train, y_train)
# print('Optimized Random Forest Classifier:')
# print(CV_clf.best_params_)

#####Optimized Random Forest
clf_final=RandomForestClassifier(max_features='auto', n_estimators= 200, max_depth=5, criterion='gini', random_state=666)
clf_final.fit(x_test,y_test)
Y_pred=clf_final.predict(x_test)
print("Accuracy Optimized Random Forest:",metrics.accuracy_score(y_test, Y_pred))