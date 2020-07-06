import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.linear_model import LogisticRegression


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
IDtest = test["PassengerId"]

train = train.drop("PassengerId", axis=1)
train = train.drop("Name", axis=1)

train.info()
train.isnull().sum()

print(train.describe())

trainMatrix = train.corr()

sn.heatmap(trainMatrix, annot=True)
##plt.save(heatmap.png)

X_train = train.drop("Survived", axis=1)
##X_train = train["Fare"]
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
##X_test  = test["Fare"]

print(X_train.shape, Y_train.shape, X_test.shape)


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)