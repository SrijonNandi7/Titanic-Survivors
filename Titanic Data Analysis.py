import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import math
titanic_data=pd.read_csv("https://raw.githubusercontent.com/pcsanwald/kaggle-titanic/master/train.csv")
titanic_data.head(10)

print("Number of passengers in current data= " + str(len(titanic_data.index)))

sns.countplot(x="survived", data=titanic_data)
sns.countplot(x="survived", hue="sex", data=titanic_data)
sns.countplot(x="survived", hue="pclass", data=titanic_data)
titanic_data["age"].plot.hist()
titanic_data["fare"].plot.hist(bins=20, figsize=(10,5))
sns.countplot(x='sibsp', data=titanic_data)


# DATA WRANGLING OR CLEANSING
titanic_data.isnull()
titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis")
sns.boxplot(x="pclass",y="age",data=titanic_data)
titanic_data.drop("cabin", axis=1, inplace=True)
titanic_data.head(5)
titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False)
titanic_data.isnull().sum()

#Changing Strings to Categorical variables
titanic_data.head(2)
pd.get_dummies(titanic_data['sex'])
sex=pd.get_dummies(titanic_data['sex'], drop_first=True)
sex.head(5)
embark=pd.get_dummies(titanic_data["embarked"], drop_first=True)
embark.head(5)
pcl=pd.get_dummies(titanic_data["pclass"], drop_first=True)
pcl.head(5)
titanic_data=pd.concat([titanic_data, sex, embark, pcl], axis=1)
titanic_data.head(5)
titanic_data.head()
titanic_data.drop(['pclass','sex','embarked','name','ticket'],axis=1,inplace=True)
titanic_data.head()

#TRAINING THE DATA
y=titanic_data["survived"]
X=titanic_data.drop('survived',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test, predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
