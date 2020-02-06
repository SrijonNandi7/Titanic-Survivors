#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math

titanic_data=pd.read_csv("https://raw.githubusercontent.com/pcsanwald/kaggle-titanic/master/train.csv")
titanic_data.head(10)


# In[102]:


print("Number of passengers in current data= " + str(len(titanic_data.index)))


# In[103]:


#ANALYZING DATA
sns.countplot(x="survived", data=titanic_data)


# In[104]:


sns.countplot(x="survived", hue="sex", data=titanic_data)


# In[105]:


sns.countplot(x="survived", hue="pclass", data=titanic_data)


# In[106]:


titanic_data["age"].plot.hist()


# In[107]:


titanic_data["fare"].plot.hist(bins=20, figsize=(10,5))


# In[108]:


sns.countplot(x='sibsp', data=titanic_data)


# In[109]:


# DATA WRANGLING OR CLEANSING

titanic_data.isnull()


# In[110]:


titanic_data.isnull().sum()


# In[111]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis")


# In[112]:


sns.boxplot(x="pclass",y="age",data=titanic_data)


# In[113]:


titanic_data.drop("cabin", axis=1, inplace=True)
titanic_data.head(5)


# In[114]:


titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False)
titanic_data.isnull().sum()


# In[115]:


#Changing Strings to Categorical variables
titanic_data.head(2)


# In[116]:


#1=true, 0=false
pd.get_dummies(titanic_data['sex'])


# In[117]:


sex=pd.get_dummies(titanic_data['sex'], drop_first=True)
sex.head(5)


# In[118]:


embark=pd.get_dummies(titanic_data["embarked"], drop_first=True)
embark.head(5)


# In[119]:


#passenger class
pcl=pd.get_dummies(titanic_data["pclass"], drop_first=True)
pcl.head(5)


# In[120]:


titanic_data=pd.concat([titanic_data, sex, embark, pcl], axis=1)
titanic_data.head(5)


# In[121]:


titanic_data.head()


# In[122]:


titanic_data.drop(['pclass','sex','embarked','name','ticket'],axis=1,inplace=True)
titanic_data.head()


# # TRAINING THE DATA

# In[123]:


y=titanic_data["survived"]
X=titanic_data.drop('survived',axis=1)


# In[129]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[130]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)


# In[131]:


predictions=logmodel.predict(X_test)


# In[132]:


from sklearn.metrics import classification_report
classification_report(y_test, predictions)


# In[134]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


# In[136]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[ ]:




