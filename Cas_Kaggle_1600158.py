#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict,GridSearchCV,StratifiedKFold,LeaveOneOut,cross_val_score,train_test_split
from sklearn.feature_selection import RFECV
import time
#from bayes_opt import BayesianOptimization, UtilityFunction

# import some data to play with
dataset = pd.read_csv("C:/Users/Usuario/OneDrive/Escritorio/student-mat.csv", header=0, delimiter=',')


# In[2]:


dataset.isnull().sum()


# In[3]:


Dataset_canvis = dataset.copy()


# In[4]:


encoder = OneHotEncoder(handle_unknown='ignore')


# # NULLS

# ## School

# In[5]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['school']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[6]:


Dataset_canvis = Dataset_canvis.rename({0: 'GP_School', 1: 'MS_School'}, axis='columns')


# In[7]:


Dataset_canvis.drop('school', axis='columns', inplace=True)


# ## Sex

# In[8]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['sex']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[9]:


Dataset_canvis = Dataset_canvis.rename({0: 'Female_Sex', 1: 'Male_Sex'}, axis='columns')


# In[10]:


Dataset_canvis.drop('sex', axis='columns', inplace=True)


# ## Address

# In[11]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['address']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[12]:


Dataset_canvis = Dataset_canvis.rename({0: 'R_Address', 1: 'U_Address'}, axis='columns')


# In[13]:


Dataset_canvis.drop('address', axis='columns', inplace=True)


# ## Famsize

# In[14]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['famsize']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[15]:


Dataset_canvis = Dataset_canvis.rename({0: 'GT3_Famsize', 1: 'LE3_Famsize'}, axis='columns')


# In[16]:


Dataset_canvis.drop('famsize', axis='columns', inplace=True)


# ## Pstatus

# In[17]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['Pstatus']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[18]:


Dataset_canvis = Dataset_canvis.rename({0: 'A_Pstatus', 1: 'T_Pstatus'}, axis='columns')


# In[19]:


Dataset_canvis.drop('Pstatus', axis='columns', inplace=True)


# ## Mjob

# In[20]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['Mjob']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[21]:


Dataset_canvis = Dataset_canvis.rename({0: 'at_home_Mjob', 1: 'health_Mjob', 2: 'other_Mjob', 3: 'services_Mjob', 4: 'teacher_Mjob' }, axis='columns')


# In[22]:


Dataset_canvis.drop('Mjob', axis='columns', inplace=True)


# ## Fjob

# In[23]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['Fjob']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[24]:


Dataset_canvis = Dataset_canvis.rename({0: 'at_home_Fjob', 1: 'health_Fjob', 2: 'other_Fjob', 3: 'services_Fjob', 4: 'teacher_Fjob' }, axis='columns')


# In[25]:


Dataset_canvis.drop('Fjob', axis='columns', inplace=True)


# ## Reason

# In[26]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['reason']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[27]:


Dataset_canvis = Dataset_canvis.rename({0: 'course_Reason', 1: 'home_Reason', 2: 'other_Reason', 3: 'reputation_Reason' }, axis='columns')


# In[28]:


Dataset_canvis.drop('reason', axis='columns', inplace=True)


# ## Guardian

# In[29]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['guardian']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[30]:


Dataset_canvis = Dataset_canvis.rename({0: 'father_Guardian', 1: 'mother_Guardian', 2: 'other_Guardian'}, axis='columns')


# In[31]:


Dataset_canvis.drop('guardian', axis='columns', inplace=True)


# ## Schoolsup

# In[32]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['schoolsup']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[33]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Schoolsup', 1: 'yes_Schoolsup'}, axis='columns')


# In[34]:


Dataset_canvis.drop('schoolsup', axis='columns', inplace=True)


# ## Famsup

# In[35]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['famsup']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[36]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Famsup', 1: 'yes_Famsup'}, axis='columns')


# In[37]:


Dataset_canvis.drop('famsup', axis='columns', inplace=True)


# ## Paid

# In[38]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['paid']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[39]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Paid', 1: 'yes_Paid'}, axis='columns')


# In[40]:


Dataset_canvis.drop('paid', axis='columns', inplace=True)


# ## Activities

# In[41]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['activities']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[42]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Activities', 1: 'yes_Activities'}, axis='columns')


# In[43]:


Dataset_canvis.drop('activities', axis='columns', inplace=True)


# ## Nursery

# In[44]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['nursery']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[45]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Nursery', 1: 'yes_Nursery'}, axis='columns')


# In[46]:


Dataset_canvis.drop('nursery', axis='columns', inplace=True)


# ## Higher

# In[47]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['higher']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[48]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Higher', 1: 'yes_Higher'}, axis='columns')


# In[49]:


Dataset_canvis.drop('higher', axis='columns', inplace=True)


# ## Internet

# In[50]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['internet']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[51]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Internet', 1: 'yes_Internet'}, axis='columns')


# In[52]:


Dataset_canvis.drop('internet', axis='columns', inplace=True)


# ## Romantic

# In[53]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['romantic']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[54]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Romantic', 1: 'yes_Romantic'}, axis='columns')


# In[55]:


Dataset_canvis.drop('romantic', axis='columns', inplace=True)


# # Variable Objectiu:

# In[56]:


X_sense_G = Dataset_canvis.copy()
X_sense_G = X_sense_G.drop(['G1','G2','G3'], axis=1)

X_amb_G = Dataset_canvis.copy()
X_amb_G = X_amb_G.drop(['G3'], axis=1)

y = Dataset_canvis.copy()
y = y[['G3']]


# In[57]:


X_sense_G_train_FS, X_sense_G_test_FS, y_sense_G_train_FS, y_sense_G_test_FS = train_test_split(X_sense_G,y,test_size=0.3,random_state = 42)


# In[58]:


X_sense_G_train_FS = StandardScaler().fit_transform(X_sense_G_train_FS)


# In[59]:


X_amb_G_train_FS, X_amb_G_test_FS, y_amb_G_train_FS, y_amb_G_test_FS = train_test_split(X_amb_G,y,test_size=0.3,random_state = 42)


# In[60]:


X_amb_G_train_FS = StandardScaler().fit_transform(X_amb_G_train_FS)


# # Feature Selection:

# ## Regressió Lasso:

# In[61]:


Lasso = Lasso()


# In[62]:


cerca_alpha_lasso = GridSearchCV(Lasso, {'alpha':np.arange(0.1,10,0.1)}, cv = 5, scoring="neg_mean_squared_error")


# ### Cerca de millors paràmetres sense G1 i G2:

# In[63]:


cerca_alpha_lasso.fit(X_sense_G_train_FS, y_sense_G_train_FS.values.ravel())


# In[64]:


print("Resultats Grid Search: \n")
print("Millor estimador dels paràmetres buscats: \n", cerca_alpha_lasso.best_estimator_)
print("Millor score dels paràmetres: \n", cerca_alpha_lasso.best_score_)
print("Millors paràmetres: \n", cerca_alpha_lasso.best_params_)


# In[65]:


coefs_sense_G = cerca_alpha_lasso.best_estimator_.coef_
importancia_sense_G = np.abs(coefs_sense_G)
importancia_sense_G


# In[66]:


X_sense_G = Dataset_canvis[np.array(X_sense_G.columns)[importancia_sense_G > 0].tolist()]


# In[67]:


correlacio_X_sense_G = X_sense_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_sense_G, annot=True, linewidths=.5)


# In[68]:


X_sense_G = X_sense_G.drop(['no_Paid'], axis=1)


# In[69]:


correlacio_X_sense_G = X_sense_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_sense_G, annot=True, linewidths=.5)


# ### Cerca de millors paràmetres amb G1 i G2:

# In[70]:


cerca_alpha_lasso.fit(X_amb_G_train_FS, y_amb_G_train_FS.values.ravel())


# In[71]:


print("Resultats Grid Search: \n")
print("Millor estimador dels paràmetres buscats: \n", cerca_alpha_lasso.best_estimator_)
print("Millor score dels paràmetres: \n", cerca_alpha_lasso.best_score_)
print("Millors paràmetres: \n", cerca_alpha_lasso.best_params_)


# In[72]:


coefs_amb_G = cerca_alpha_lasso.best_estimator_.coef_
importancia_amb_G = np.abs(coefs_amb_G)
importancia_amb_G


# In[73]:


X_amb_G = Dataset_canvis[np.array(X_amb_G.columns)[importancia_amb_G > 0].tolist()]


# In[74]:


correlacio_X_amb_G = X_amb_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_amb_G, annot=True, linewidths=.5)


# In[75]:


X_amb_G = X_amb_G.drop(['no_Schoolsup', 'no_Paid', 'no_Activities', 'no_Romantic', 'other_Fjob', 'no_Higher','G1'], axis=1)


# In[76]:


correlacio_X_amb_G = X_amb_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_amb_G, annot=True, linewidths=.5)


# ## Regressió RandomForest

# In[77]:


rf = RandomForestRegressor(random_state=0)


# In[78]:


grid_param = {
    'n_estimators':np.arange(80,150,10),
    'criterion':['squared_error','absolute_error','poisson'],
    'max_depth': np.arange(2,20,1), 
    'min_samples_leaf': np.arange(1,10,1),
    'min_samples_split': np.arange(2,10,1),
    'max_features':['sqrt', 'log2', 'auto', 1.0]
}


# In[79]:


cerca_params_rf = GridSearchCV(rf, grid_param, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")


# ### Cerca de millors paràmetres sense G1 i G2:

# In[80]:


#cerca_params_rf.fit(X_sense_G_train, y_sense_G_train.values.ravel())


# ### Cerca de millors paràmetres amb G1 i G2:

# In[81]:


#cerca_params_rf.fit(X_amb_G_train_SS, y_amb_G_train.values.ravel())

