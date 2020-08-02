#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np


# In[62]:


df= pd.read_csv("train.csv")
df.head()


# In[63]:


df.isnull().sum()


# In[4]:


sample_sub = pd.read_csv('sample_submission.csv')
sample_sub.head()


# In[5]:


df['SalePrice'].describe()


# In[7]:


import matplotlib.pyplot as plt

df['SalePrice'].hist()


# In[10]:


df.corr()


# In[12]:


import seaborn as sns
corrmat = df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[13]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[64]:


selected_col=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
df[selected_col].isnull().sum()


# In[65]:


x_data = df[selected_col].values
y_data = df['SalePrice'].values


# In[66]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=6)


# <h3> Linear Regression </h3>

# In[67]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[68]:


model=lr.fit(x_train,y_train)


# In[69]:


print(model.intercept_)
print(model.coef_)


# In[74]:


yhat=model.predict(x_test)

HousingTestingData =pd.DataFrame(data=x_test, columns=selected_col)
HousingTestingData['SalePrice']=y_test
HousingTestingData[('Predicted SalePrice')]=yhat
HousingTestingData.head()


# In[77]:


from sklearn import metrics
 
#Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, model.predict(x_train)))
 
#Measuring accuracy on Testing Data
print('Accuracy',100- (np.mean(np.abs((y_test - yhat) / y_test)) * 100))


# <h3> Decision Tree Regressor </h3>

# In[78]:


from sklearn.tree import DecisionTreeRegressor
RegModel= DecisionTreeRegressor(max_depth=3 , criterion = 'mse')
print(RegModel)


# In[80]:


DTree= RegModel.fit(x_train,y_train)
prediction_tree = DTree.predict(x_test)
#print(prediction_tree)


# In[82]:


from sklearn import metrics
 
#Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, RegModel.predict(x_train)))
 
#Measuring accuracy on Testing Data
print('Accuracy',100- (np.mean(np.abs((y_test - prediction_tree) / y_test)) * 100))
#100-MAPE


# <h3> Random Forest Regressor </h3>

# In[83]:


from sklearn.ensemble import RandomForestRegressor
RegModel= RandomForestRegressor(n_estimators=100,criterion='mse')
print(RegModel)


# In[85]:


RF=RegModel.fit(x_train,y_train)
prediction_RF=RF.predict(x_test)
#prediction_RF


# In[86]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

feature_importances = pd.Series(RF.feature_importances_, index=selected_col)
feature_importances.nlargest(10).plot(kind='barh')


# In[88]:


from sklearn import metrics
print('R2 Value:',metrics.r2_score(y_train, RF.predict(x_train)))
 
#Measuring accuracy on Testing Data
print('Accuracy',100- (np.mean(np.abs((y_test - prediction_RF) / y_test)) * 100))


# <h3> XG Boost Regressor </h3>

# In[89]:


from xgboost import XGBRegressor
RegModel=XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=500, objective='reg:linear', booster='gbtree')
 
print(RegModel)


# In[90]:


XGB=RegModel.fit(x_train,y_train)
prediction_XGB=XGB.predict(x_test)


# In[91]:


#Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')

feature_importances = pd.Series(XGB.feature_importances_, index=selected_col)
feature_importances.nlargest(10).plot(kind='barh')


# In[93]:


#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y_train, XGB.predict(x_train)))
 
#Measuring accuracy on Testing Data
print('Accuracy',100- (np.mean(np.abs((y_test - prediction_XGB) / y_test)) * 100))
 


# <h3> Based on the above algorithm's accuracy , we are choosing Random Forest Regression </h3>

# Based on the Feature Importance matrix, we are choosign the top 3 predictors 

# In[94]:


new_selected_col=['OverallQual','GrLivArea','GarageCars']
x_data = df[new_selected_col].values
y_data = df['SalePrice'].values


# In[95]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=6)


# In[96]:


from sklearn.ensemble import RandomForestRegressor
RegModel= RandomForestRegressor(n_estimators=100,criterion='mse')
print(RegModel)


# In[97]:


RF=RegModel.fit(x_train,y_train)
prediction_RF=RF.predict(x_test)


# In[105]:


HousingTrainingData =pd.DataFrame(data=x_test, columns=new_selected_col)
HousingTrainingData['SalePrice']=y_test
HousingTrainingData[('Predicted SalePrice')]=prediction_RF
HousingTrainingData.head()


# In[100]:


from sklearn import metrics
 
#Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, RF.predict(x_train)))
 
#Measuring accuracy on Testing Data
print('Accuracy',100- (np.mean(np.abs((y_test - prediction_RF) / y_test)) * 100))


# <h4> Test Data given </h4>

# In[101]:


df_test=pd.read_csv("test.csv")
df_test.head()


# In[102]:


df_test[new_selected_col].isnull().sum()


# In[103]:


df_test['GarageCars'].fillna(0,inplace=True)


# In[104]:


df_test[new_selected_col].isnull().sum()


# In[106]:


df_test[new_selected_col].head()


# In[107]:


X = df_test[new_selected_col].values


# In[109]:


# Predicting the SalePrice values for testing data
Pred = RegModel.predict(X)

# Creating a DataFrame
HousingTestingData=pd.DataFrame(df_test, columns=new_selected_col)
HousingTestingData['Predicted SalePrice']=Pred
HousingTestingData.head()


# In[111]:


df_test['Predicted SalePrice']=Pred


# In[112]:


df_test.head()


# In[116]:


submission = df_test[['Id','Predicted SalePrice']]


# In[117]:


submission.head()


# In[118]:


submission.to_csv("mysubmission.csv")

