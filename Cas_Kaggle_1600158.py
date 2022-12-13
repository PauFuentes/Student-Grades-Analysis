#!/usr/bin/env python
# coding: utf-8

# In[162]:


import math
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict,GridSearchCV, ShuffleSplit, KFold,LeaveOneOut,cross_val_score,train_test_split
from sklearn.feature_selection import RFECV
from catboost import CatBoostRegressor, Pool
from xgboost.sklearn import XGBRegressor
from imblearn.over_sampling import SMOTE
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


# ## Distribució atribut G3:

# In[56]:


plt.figure()
plt.title("Histograma de l'atribut G3")
plt.xlabel("Attribute Value")
plt.ylabel("Count")
hist = plt.hist(dataset['G3'], bins=11, range=[np.min(dataset['G3']), np.max(dataset['G3'])], histtype="bar", rwidth=0.8)


# # Variable Objectiu:

# In[57]:


X_sense_G = Dataset_canvis.copy()
X_sense_G = X_sense_G.drop(['G1','G2','G3'], axis=1)

X_amb_G = Dataset_canvis.copy()
X_amb_G = X_amb_G.drop(['G3'], axis=1)

y = Dataset_canvis.copy()
y = y['G3']


# In[58]:


X_sense_G_train_FS, X_sense_G_test_FS, y_sense_G_train_FS, y_sense_G_test_FS = train_test_split(X_sense_G,y,test_size=0.3,random_state = 42)


# In[59]:


X_sense_G_train_FS = StandardScaler().fit_transform(X_sense_G_train_FS)


# In[60]:


X_amb_G_train_FS, X_amb_G_test_FS, y_amb_G_train_FS, y_amb_G_test_FS = train_test_split(X_amb_G,y,test_size=0.3,random_state = 42)


# In[61]:


X_amb_G_train_FS = StandardScaler().fit_transform(X_amb_G_train_FS)


# # Feature Selection:

# ## Regressió Lasso:

# In[62]:


Lasso = Lasso()


# In[63]:


cerca_alpha_lasso = GridSearchCV(Lasso, {'alpha':np.arange(0.1,10,0.1)}, cv = 5, scoring="neg_mean_squared_error")


# ### Cerca de millors paràmetres sense G1 i G2:

# In[64]:


cerca_alpha_lasso.fit(X_sense_G_train_FS, y_sense_G_train_FS.values.ravel())


# In[65]:


print("Resultats Grid Search: \n")
print("Millor estimador dels paràmetres buscats: \n", cerca_alpha_lasso.best_estimator_)
print("Millor score dels paràmetres: \n", cerca_alpha_lasso.best_score_)
print("Millors paràmetres: \n", cerca_alpha_lasso.best_params_)


# In[66]:


coefs_sense_G = cerca_alpha_lasso.best_estimator_.coef_
importancia_sense_G = np.abs(coefs_sense_G)
importancia_sense_G


# In[67]:


X_sense_G = Dataset_canvis[np.array(X_sense_G.columns)[importancia_sense_G > 0].tolist()]


# In[68]:


correlacio_X_sense_G = X_sense_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_sense_G, annot=True, linewidths=.5)


# In[69]:


X_sense_G = X_sense_G.drop(['no_Paid'], axis=1)


# In[70]:


correlacio_X_sense_G = X_sense_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_sense_G, annot=True, linewidths=.5)


# ### Cerca de millors paràmetres amb G1 i G2:

# In[71]:


cerca_alpha_lasso.fit(X_amb_G_train_FS, y_amb_G_train_FS.values.ravel())


# In[72]:


print("Resultats Grid Search: \n")
print("Millor estimador dels paràmetres buscats: \n", cerca_alpha_lasso.best_estimator_)
print("Millor score dels paràmetres: \n", cerca_alpha_lasso.best_score_)
print("Millors paràmetres: \n", cerca_alpha_lasso.best_params_)


# In[73]:


coefs_amb_G = cerca_alpha_lasso.best_estimator_.coef_
importancia_amb_G = np.abs(coefs_amb_G)
importancia_amb_G


# In[74]:


X_amb_G = Dataset_canvis[np.array(X_amb_G.columns)[importancia_amb_G > 0].tolist()]


# In[75]:


correlacio_X_amb_G = X_amb_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_amb_G, annot=True, linewidths=.5)


# In[76]:


X_amb_G = X_amb_G.drop(['no_Schoolsup', 'no_Paid', 'no_Activities', 'no_Romantic', 'other_Fjob', 'no_Higher','G1'], axis=1)


# In[77]:


correlacio_X_amb_G = X_amb_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_amb_G, annot=True, linewidths=.5)


# ## Regressió RandomForest

# In[78]:


rf = RandomForestRegressor(random_state=0)


# In[79]:


grid_param = {
    'n_estimators':np.arange(80,150,10),
    'criterion':['squared_error','absolute_error','poisson'],
    'max_depth': np.arange(2,20,1), 
    'min_samples_leaf': np.arange(1,10,1),
    'min_samples_split': np.arange(2,10,1),
    'max_features':['sqrt', 'log2', 'auto', 1.0]
}


# In[80]:


cerca_params_rf = GridSearchCV(rf, grid_param, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")


# ### Cerca de millors paràmetres sense G1 i G2:

# In[81]:


#cerca_params_rf.fit(X_sense_G_train, y_sense_G_train.values.ravel())


# ### Cerca de millors paràmetres amb G1 i G2:

# In[82]:


#cerca_params_rf.fit(X_amb_G_train_SS, y_amb_G_train.values.ravel())


# # Regressions:

# In[83]:


Dic_MSE_Sense_G = {}
Dic_MSE_Amb_G = {}

Dic_r2_Sense_G = {}
Dic_r2_Amb_G = {}


# In[84]:


X_sense_G_train, X_sense_G_test, y_sense_G_train, y_sense_G_test = train_test_split(X_sense_G,y,test_size=0.3,random_state = 23)


# In[85]:


X_sense_G_train = StandardScaler().fit_transform(X_sense_G_train)
X_sense_G_test = StandardScaler().fit_transform(X_sense_G_test)


# In[86]:


X_amb_G_train, X_amb_G_test, y_amb_G_train, y_amb_G_test = train_test_split(X_amb_G,y,test_size=0.3,random_state = 50)


# In[87]:


X_amb_G_train = StandardScaler().fit_transform(X_amb_G_train)
X_amb_G_test = StandardScaler().fit_transform(X_amb_G_test)


# ## Regressió linial sense G1 ni G2:

# In[88]:


LinReg = LinearRegression()


# In[89]:


LinReg.fit(X_sense_G_train, y_sense_G_train)


# In[90]:


Pred_LinReg_sense_G = LinReg.predict(X_sense_G_test)


# ## Regressió linial amb G1 i G2:

# In[91]:


LinReg.fit(X_amb_G_train, y_amb_G_train)


# In[92]:


Pred_LinReg_amb_G = LinReg.predict(X_amb_G_test)


# In[93]:


# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
plt.figure()
ax = sns.regplot(x = y_sense_G_test,y = Pred_LinReg_sense_G, fit_reg=True, color = 'green')#plt.scatter(y_sense_G_test, Pred_CBR_sense_G)
ax2 = sns.regplot(x = y_amb_G_test,y = Pred_LinReg_amb_G, fit_reg=True, color = 'red')
plt.ylabel("Valors predits")
plt.xlabel("Valors reals")
plt.title("Linear Regression")
# Mostrem l'error (MSE i R2)
MSE_LinReg_sense_G = mean_squared_error(y_sense_G_test, Pred_LinReg_sense_G)
MSE_LinReg_amb_G = mean_squared_error(y_amb_G_test, Pred_LinReg_amb_G)

r2_LinReg_sense_G = r2_score(y_sense_G_test, Pred_LinReg_sense_G)
r2_LinReg_amb_G = r2_score(y_amb_G_test, Pred_LinReg_amb_G)


print("Mean squeared error sense G1 ni G2: ", MSE_LinReg_sense_G)
print("Mean squeared error amb G1 i G2: ", MSE_LinReg_amb_G)

print("R2 score sense G1 ni G2: ", r2_LinReg_sense_G)
print("R2 score amb G1 i G2: ", r2_LinReg_amb_G)

Dic_MSE_Sense_G['LinReg'] = MSE_LinReg_sense_G
Dic_MSE_Amb_G['LinReg'] = MSE_LinReg_amb_G

Dic_r2_Sense_G['LinReg'] = r2_LinReg_sense_G
Dic_r2_Amb_G['LinReg'] = r2_LinReg_amb_G


# ## SVR sense G1 i G2:

# In[94]:


Svr = SVR()


# In[95]:


Svr.fit(X_sense_G_train, y_sense_G_train.values.ravel())


# In[96]:


Pred_SVR_sense_G = Svr.predict(X_sense_G_test)


# ## SVR amb G1 i G2:

# In[97]:


Svr.fit(X_amb_G_train, y_amb_G_train.values.ravel())


# In[98]:


Pred_SVR_amb_G = Svr.predict(X_amb_G_test)


# In[99]:


# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
plt.figure()
ax = sns.regplot(x = y_sense_G_test,y = Pred_SVR_sense_G, fit_reg=True, color = 'green')#plt.scatter(y_sense_G_test, Pred_CBR_sense_G)
ax2 = sns.regplot(x = y_amb_G_test,y = Pred_SVR_amb_G, fit_reg=True, color = 'red')
plt.ylabel("Valors predits")
plt.xlabel("Valors reals")
plt.title("SVR")
# Mostrem l'error (MSE i R2)
MSE_SVR_sense_G = mean_squared_error(y_sense_G_test, Pred_SVR_sense_G)
MSE_SVR_amb_G = mean_squared_error(y_amb_G_test, Pred_SVR_amb_G)

r2_SVR_sense_G = r2_score(y_sense_G_test, Pred_SVR_sense_G)
r2_SVR_amb_G = r2_score(y_amb_G_test, Pred_SVR_amb_G)


print("Mean squeared error sense G1 ni G2: ", MSE_SVR_sense_G)
print("Mean squeared error amb G1 i G2: ", MSE_SVR_amb_G)

print("R2 score sense G1 ni G2: ", r2_SVR_sense_G)
print("R2 score amb G1 i G2: ", r2_SVR_amb_G)

Dic_MSE_Sense_G['SVR'] = MSE_SVR_sense_G
Dic_MSE_Amb_G['SVR'] = MSE_SVR_amb_G

Dic_r2_Sense_G['SVR'] = r2_SVR_sense_G
Dic_r2_Amb_G['SVR'] = r2_SVR_amb_G


# ## BayesianRidge sense G1 i G2:

# In[100]:


BR = BayesianRidge()


# In[101]:


BR.fit(X_sense_G_train, y_sense_G_train.values.ravel())


# In[102]:


Pred_BR_sense_G = BR.predict(X_sense_G_test)


# ## BayesianRidge amb G1 i G2:

# In[103]:


BR.fit(X_amb_G_train, y_amb_G_train.values.ravel())


# In[104]:


Pred_BR_amb_G = BR.predict(X_amb_G_test)


# In[105]:


# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
plt.figure()
ax = sns.regplot(x = y_sense_G_test,y = Pred_BR_sense_G, fit_reg=True, color = 'green')#plt.scatter(y_sense_G_test, Pred_CBR_sense_G)
ax2 = sns.regplot(x = y_amb_G_test,y = Pred_BR_amb_G, fit_reg=True, color = 'red')
plt.ylabel("Valors predits")
plt.xlabel("Valors reals")
plt.title("BayesianRidge")
# Mostrem l'error (MSE i R2)
MSE_BR_sense_G = mean_squared_error(y_sense_G_test, Pred_BR_sense_G)
MSE_BR_amb_G = mean_squared_error(y_amb_G_test, Pred_BR_amb_G)

r2_BR_sense_G = r2_score(y_sense_G_test, Pred_BR_sense_G)
r2_BR_amb_G = r2_score(y_amb_G_test, Pred_BR_amb_G)


print("Mean squeared error sense G1 ni G2: ", MSE_BR_sense_G)
print("Mean squeared error amb G1 i G2: ", MSE_BR_amb_G)

print("R2 score sense G1 ni G2: ", r2_BR_sense_G)
print("R2 score amb G1 i G2: ", r2_BR_amb_G)

Dic_MSE_Sense_G['BR'] = MSE_BR_sense_G
Dic_MSE_Amb_G['BR'] = MSE_BR_amb_G

Dic_r2_Sense_G['BR'] = r2_BR_sense_G
Dic_r2_Amb_G['BR'] = r2_BR_amb_G


# ## Regressió CatBoost sense G1 i G2:

# In[157]:


CBR = CatBoostRegressor(verbose = None, silent = True)


# In[107]:


CBR.fit(X_sense_G_train, y_sense_G_train.values.ravel(), verbose = False)


# In[108]:


Pred_CBR_sense_G = CBR.predict(X_sense_G_test)


# ## Regressió CatBoost amb G1 i G2:

# In[109]:


CBR.fit(X_amb_G_train, y_amb_G_train.values.ravel(), verbose = False)


# In[110]:


Pred_CBR_amb_G = CBR.predict(X_amb_G_test)


# In[111]:


# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
plt.figure()
ax = sns.regplot(x = y_sense_G_test,y = Pred_CBR_sense_G, fit_reg=True, color = 'green')#plt.scatter(y_sense_G_test, Pred_CBR_sense_G)
ax2 = sns.regplot(x = y_amb_G_test,y = Pred_CBR_amb_G, fit_reg=True, color = 'red')
plt.ylabel("Valors predits")
plt.xlabel("Valors reals")
plt.title("CatBoostRegressor")
# Mostrem l'error (MSE i R2)
MSE_CBR_sense_G = mean_squared_error(y_sense_G_test, Pred_CBR_sense_G)
MSE_CBR_amb_G = mean_squared_error(y_amb_G_test, Pred_CBR_amb_G)

r2_CBR_sense_G = r2_score(y_sense_G_test, Pred_CBR_sense_G)
r2_CBR_amb_G = r2_score(y_amb_G_test, Pred_CBR_amb_G)


print("Mean squeared error sense G1 ni G2: ", MSE_CBR_sense_G)
print("Mean squeared error amb G1 i G2: ", MSE_CBR_amb_G)

print("R2 score sense G1 ni G2: ", r2_CBR_sense_G)
print("R2 score amb G1 i G2: ", r2_CBR_amb_G)

Dic_MSE_Sense_G['CBR'] = MSE_CBR_sense_G
Dic_MSE_Amb_G['CBR'] = MSE_CBR_amb_G

Dic_r2_Sense_G['CBR'] = r2_CBR_sense_G
Dic_r2_Amb_G['CBR'] = r2_CBR_amb_G


# ## Regressió XGBoost sense G1 i G2:

# In[112]:


XGB = XGBRegressor()


# In[113]:


XGB.fit(X_sense_G_train, y_sense_G_train)


# In[114]:


Pred_XGB_sense_G = XGB.predict(X_sense_G_test)


# ## Regressió XGBoost amb G1 i G2:

# In[115]:


XGB.fit(X_amb_G_train, y_amb_G_train)


# In[116]:


Pred_XGB_amb_G = XGB.predict(X_amb_G_test)


# In[117]:


# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
plt.figure()
ax = sns.regplot(x = y_sense_G_test,y = Pred_XGB_sense_G, fit_reg=True, color = 'green')#plt.scatter(y_sense_G_test, Pred_CBR_sense_G)
ax2 = sns.regplot(x = y_amb_G_test,y = Pred_XGB_amb_G, fit_reg=True, color = 'red')
plt.ylabel("Valors predits")
plt.xlabel("Valors reals")
plt.title("XGBoost Regressor")
# Mostrem l'error (MSE i R2)
MSE_XGB_sense_G = mean_squared_error(y_sense_G_test, Pred_XGB_sense_G)
MSE_XGB_amb_G = mean_squared_error(y_amb_G_test, Pred_XGB_amb_G)

r2_XGB_sense_G = r2_score(y_sense_G_test, Pred_XGB_sense_G)
r2_XGB_amb_G = r2_score(y_amb_G_test, Pred_XGB_amb_G)


print("Mean squeared error sense G1 ni G2: ", MSE_XGB_sense_G)
print("Mean squeared error amb G1 i G2: ", MSE_XGB_amb_G)

print("R2 score sense G1 ni G2: ", r2_XGB_sense_G)
print("R2 score amb G1 i G2: ", r2_XGB_amb_G)

Dic_MSE_Sense_G['XGB'] = MSE_XGB_sense_G
Dic_MSE_Amb_G['XGB'] = MSE_XGB_amb_G

Dic_r2_Sense_G['XGB'] = r2_XGB_sense_G
Dic_r2_Amb_G['XGB'] = r2_XGB_amb_G


# In[118]:


print("Model amb MSE més baix sense G1 ni G2: ", min(Dic_MSE_Sense_G, key=Dic_MSE_Sense_G.get), "---->",Dic_MSE_Sense_G[min(Dic_MSE_Sense_G, key=Dic_MSE_Sense_G.get)] )
print("Model amb r2 score més alta sense G1 ni G2: ", max(Dic_r2_Sense_G, key=Dic_r2_Sense_G.get), "---->",Dic_r2_Sense_G[max(Dic_r2_Sense_G, key=Dic_r2_Sense_G.get)] )
print("Model amb MSE més baix amb G1 i G2: ", min(Dic_MSE_Amb_G, key=Dic_MSE_Amb_G.get), "---->",Dic_MSE_Amb_G[min(Dic_MSE_Amb_G, key=Dic_MSE_Amb_G.get)])
print("Model amb r2 score més alta amb G1 i G2: ", max(Dic_r2_Amb_G, key=Dic_r2_Amb_G.get), "---->",Dic_r2_Amb_G[max(Dic_r2_Amb_G, key=Dic_r2_Amb_G.get)])


# # Cross-Validation:

# In[119]:


X_sense_G_cross = Dataset_canvis.copy()
X_sense_G_cross = X_sense_G_cross.drop(['G1','G2','G3'], axis=1)
X_sense_G_cross = StandardScaler().fit_transform(X_sense_G_cross)

X_amb_G_cross = Dataset_canvis.copy()
X_amb_G_cross = X_amb_G_cross.drop(['G3'], axis=1)
X_amb_G_cross = StandardScaler().fit_transform(X_amb_G_cross)

y_cross = Dataset_canvis.copy()
y_cross = y_cross[['G3']]


# ## CV sense G1 ni G2:

# In[120]:


SS_3 = ShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 0)
SS_5 = ShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 0)
SS_10 = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 0)

loo = LeaveOneOut()


# ### Regressió linial (3, 5 i 10 folds i LeaveOneOut):

# ##### Sense G1 ni G2:

# In[121]:


scores3_LinReg_sense_G = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_LinReg_sense_G), np.std(scores3_LinReg_sense_G)))


# In[122]:


scores5_LinReg_sense_G = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_LinReg_sense_G), np.std(scores5_LinReg_sense_G)))


# In[123]:


scores10_LinReg_sense_G = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_LinReg_sense_G), np.std(scores10_LinReg_sense_G)))


# In[127]:


scoresloo_LinReg_sense_G = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_LinReg_sense_G), np.std(scoresloo_LinReg_sense_G)))


# ##### Amb G1  i G2:

# In[135]:


scores3_LinReg_amb_G = cross_val_score(LinReg, X_amb_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_LinReg_amb_G), np.std(scores3_LinReg_amb_G)))


# In[136]:


scores5_LinReg_amb_G = cross_val_score(LinReg, X_amb_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_LinReg_amb_G), np.std(scores5_LinReg_amb_G)))


# In[137]:


scores10_LinReg_amb_G = cross_val_score(LinReg, X_amb_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_LinReg_amb_G), np.std(scores10_LinReg_amb_G)))


# In[138]:


scoresloo_LinReg_amb_G = cross_val_score(LinReg, X_amb_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_LinReg_amb_G), np.std(scoresloo_LinReg_amb_G)))


# ### SVR (3, 5 i 10 folds i LeaveOneOut):

# #### Sense G1 ni G2:

# In[128]:


scores3_SVR_sense_G = cross_val_score(Svr, X_sense_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_SVR_sense_G), np.std(scores3_SVR_sense_G)))


# In[132]:


scores5_SVR_sense_G = cross_val_score(Svr, X_sense_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_SVR_sense_G), np.std(scores5_SVR_sense_G)))


# In[133]:


scores10_SVR_sense_G = cross_val_score(Svr, X_sense_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_SVR_sense_G), np.std(scores10_SVR_sense_G)))


# In[134]:


scoresloo_SVR_sense_G = cross_val_score(Svr, X_sense_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_SVR_sense_G), np.std(scoresloo_SVR_sense_G)))


# #### Amb G1 i G2:

# In[139]:


scores3_SVR_amb_G = cross_val_score(Svr, X_amb_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_SVR_amb_G), np.std(scores3_SVR_amb_G)))


# In[140]:


scores5_SVR_amb_G = cross_val_score(Svr, X_amb_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_SVR_amb_G), np.std(scores5_SVR_amb_G)))


# In[141]:


scores10_SVR_amb_G = cross_val_score(Svr, X_amb_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_SVR_amb_G), np.std(scores10_SVR_amb_G)))


# In[143]:


scoresloo_SVR_amb_G = cross_val_score(Svr, X_amb_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_SVR_amb_G), np.std(scoresloo_SVR_amb_G)))


# ### BayesianRidge (3, 5 i 10 folds i LeaveOneOut):

# #### Sense G1 ni G2:

# In[144]:


scores3_BR_sense_G = cross_val_score(BR, X_sense_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_BR_sense_G), np.std(scores3_BR_sense_G)))


# In[145]:


scores5_BR_sense_G = cross_val_score(BR, X_sense_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_BR_sense_G), np.std(scores5_BR_sense_G)))


# In[146]:


scores10_BR_sense_G = cross_val_score(BR, X_sense_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_BR_sense_G), np.std(scores10_BR_sense_G)))


# In[147]:


scoresloo_BR_sense_G = cross_val_score(BR, X_sense_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_BR_sense_G), np.std(scoresloo_BR_sense_G)))


# #### Amb G1 i G2:

# In[149]:


scores3_BR_amb_G = cross_val_score(BR, X_amb_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_BR_amb_G), np.std(scores3_BR_amb_G)))


# In[150]:


scores5_BR_amb_G = cross_val_score(BR, X_amb_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_BR_amb_G), np.std(scores5_BR_amb_G)))


# In[151]:


scores10_BR_amb_G = cross_val_score(BR, X_amb_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_BR_amb_G), np.std(scores10_BR_amb_G)))


# In[152]:


scoresloo_BR_amb_G = cross_val_score(BR, X_amb_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_BR_amb_G), np.std(scoresloo_BR_amb_G)))


# ### CatBoost (3, 5 i 10 folds i LeaveOneOut):

# #### Sense G1 ni G2:

# In[158]:


scores3_CBR_sense_G = cross_val_score(CBR, X_sense_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_CBR_sense_G), np.std(scores3_CBR_sense_G)))


# In[159]:


scores5_CBR_sense_G = cross_val_score(CBR, X_sense_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_CBR_sense_G), np.std(scores5_CBR_sense_G)))


# In[160]:


scores10_CBR_sense_G = cross_val_score(CBR, X_sense_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_CBR_sense_G), np.std(scores10_CBR_sense_G)))


# In[172]:


scoresloo_CBR_sense_G = cross_val_score(CBR, X_sense_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_CBR_sense_G), np.std(scoresloo_CBR_sense_G)))


# #### Amb G1 i G2:

# In[173]:


scores3_CBR_amb_G = cross_val_score(CBR, X_amb_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_CBR_amb_G), np.std(scores3_CBR_amb_G)))


# In[174]:


scores5_CBR_amb_G = cross_val_score(CBR, X_amb_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_CBR_amb_G), np.std(scores5_CBR_amb_G)))


# In[175]:


scores10_CBR_amb_G = cross_val_score(CBR, X_amb_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_CBR_amb_G), np.std(scores10_CBR_amb_G)))


# In[176]:


scoresloo_CBR_amb_G = cross_val_score(CBR, X_amb_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_CBR_amb_G), np.std(scoresloo_CBR_amb_G)))


# ### XGBoost (3, 5 i 10 folds i LeaveOneOut):

# #### Sense G1 ni G2:

# In[177]:


scores3_XGB_sense_G = cross_val_score(XGB, X_sense_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_XGB_sense_G), np.std(scores3_XGB_sense_G)))


# In[178]:


scores5_XGB_sense_G = cross_val_score(XGB, X_sense_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_XGB_sense_G), np.std(scores5_XGB_sense_G)))


# In[179]:


scores10_XGB_sense_G = cross_val_score(XGB, X_sense_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_XGB_sense_G), np.std(scores10_XGB_sense_G)))


# In[180]:


scoresloo_XGB_sense_G = cross_val_score(XGB, X_sense_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_XGB_sense_G), np.std(scoresloo_XGB_sense_G)))


# #### Amb G1 i G2:

# In[181]:


scores3_XGB_amb_G = cross_val_score(XGB, X_amb_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_XGB_amb_G), np.std(scores3_XGB_amb_G)))


# In[182]:


scores5_XGB_amb_G = cross_val_score(XGB, X_amb_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_XGB_amb_G), np.std(scores5_XGB_amb_G)))


# In[183]:


scores10_XGB_amb_G = cross_val_score(XGB, X_amb_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_XGB_amb_G), np.std(scores10_XGB_amb_G)))


# In[184]:


scoresloo_XGB_amb_G = cross_val_score(XGB, X_amb_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_XGB_amb_G), np.std(scoresloo_XGB_amb_G)))

