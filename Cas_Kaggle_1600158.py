#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from imblearn.over_sampling import SMOTE
import time
#from bayes_opt import BayesianOptimization, UtilityFunction

# import some data to play with
dataset = pd.read_csv("/home/pau/Desktop/Tercer Curs/Primer Semestre/AComp/Cas_Kaggle/student-mat.csv", header=0, delimiter=',')


# In[ ]:


dataset.isnull().sum()


# In[ ]:


Dataset_canvis = dataset.copy()


# In[ ]:


encoder = OneHotEncoder(handle_unknown='ignore')


# # NULLS

# ## School

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['school']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'GP_School', 1: 'MS_School'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('school', axis='columns', inplace=True)


# ## Sex

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['sex']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'Female_Sex', 1: 'Male_Sex'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('sex', axis='columns', inplace=True)


# ## Address

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['address']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'R_Address', 1: 'U_Address'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('address', axis='columns', inplace=True)


# ## Famsize

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['famsize']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'GT3_Famsize', 1: 'LE3_Famsize'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('famsize', axis='columns', inplace=True)


# ## Pstatus

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['Pstatus']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'A_Pstatus', 1: 'T_Pstatus'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('Pstatus', axis='columns', inplace=True)


# ## Mjob

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['Mjob']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'at_home_Mjob', 1: 'health_Mjob', 2: 'other_Mjob', 3: 'services_Mjob', 4: 'teacher_Mjob' }, axis='columns')


# In[ ]:


Dataset_canvis.drop('Mjob', axis='columns', inplace=True)


# ## Fjob

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['Fjob']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'at_home_Fjob', 1: 'health_Fjob', 2: 'other_Fjob', 3: 'services_Fjob', 4: 'teacher_Fjob' }, axis='columns')


# In[ ]:


Dataset_canvis.drop('Fjob', axis='columns', inplace=True)


# ## Reason

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['reason']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'course_Reason', 1: 'home_Reason', 2: 'other_Reason', 3: 'reputation_Reason' }, axis='columns')


# In[ ]:


Dataset_canvis.drop('reason', axis='columns', inplace=True)


# ## Guardian

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['guardian']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'father_Guardian', 1: 'mother_Guardian', 2: 'other_Guardian'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('guardian', axis='columns', inplace=True)


# ## Schoolsup

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['schoolsup']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Schoolsup', 1: 'yes_Schoolsup'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('schoolsup', axis='columns', inplace=True)


# ## Famsup

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['famsup']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Famsup', 1: 'yes_Famsup'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('famsup', axis='columns', inplace=True)


# ## Paid

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['paid']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Paid', 1: 'yes_Paid'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('paid', axis='columns', inplace=True)


# ## Activities

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['activities']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Activities', 1: 'yes_Activities'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('activities', axis='columns', inplace=True)


# ## Nursery

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['nursery']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Nursery', 1: 'yes_Nursery'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('nursery', axis='columns', inplace=True)


# ## Higher

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['higher']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Higher', 1: 'yes_Higher'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('higher', axis='columns', inplace=True)


# ## Internet

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['internet']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Internet', 1: 'yes_Internet'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('internet', axis='columns', inplace=True)


# ## Romantic

# In[ ]:


encoder_df = pd.DataFrame(encoder.fit_transform(Dataset_canvis[['romantic']]).toarray())
Dataset_canvis = Dataset_canvis.join(encoder_df)


# In[ ]:


Dataset_canvis = Dataset_canvis.rename({0: 'no_Romantic', 1: 'yes_Romantic'}, axis='columns')


# In[ ]:


Dataset_canvis.drop('romantic', axis='columns', inplace=True)


# ## Distribució atribut G3:

# In[ ]:


plt.figure()
plt.title("Histograma de l'atribut G3")
plt.xlabel("Attribute Value")
plt.ylabel("Count")
hist = plt.hist(dataset['G3'], bins=11, range=[np.min(dataset['G3']), np.max(dataset['G3'])], histtype="bar", rwidth=0.8)


# # Variable Objectiu:

# In[ ]:


X_sense_G = Dataset_canvis.copy()
X_sense_G = X_sense_G.drop(['G1','G2','G3'], axis=1)

X_amb_G = Dataset_canvis.copy()
X_amb_G = X_amb_G.drop(['G3'], axis=1)

y = Dataset_canvis.copy()
y = y['G3']


# In[ ]:


X_sense_G_train_FS, X_sense_G_test_FS, y_sense_G_train_FS, y_sense_G_test_FS = train_test_split(X_sense_G,y,test_size=0.3,random_state = 42)


# In[ ]:


X_sense_G_train_FS = StandardScaler().fit_transform(X_sense_G_train_FS)


# In[ ]:


X_amb_G_train_FS, X_amb_G_test_FS, y_amb_G_train_FS, y_amb_G_test_FS = train_test_split(X_amb_G,y,test_size=0.3,random_state = 42)


# In[ ]:


X_amb_G_train_FS = StandardScaler().fit_transform(X_amb_G_train_FS)


# # Feature Selection:

# ## Regressió Lasso:

# In[ ]:


Lasso = Lasso()


# In[ ]:


cerca_alpha_lasso = GridSearchCV(Lasso, {'alpha':np.arange(0.1,10,0.1)}, cv = 5, scoring="neg_mean_squared_error")


# ### Cerca de millors paràmetres sense G1 i G2:

# In[ ]:


cerca_alpha_lasso.fit(X_sense_G_train_FS, y_sense_G_train_FS.values.ravel())


# In[ ]:


print("Resultats Grid Search: \n")
print("Millor estimador dels paràmetres buscats: \n", cerca_alpha_lasso.best_estimator_)
print("Millor score dels paràmetres: \n", cerca_alpha_lasso.best_score_)
print("Millors paràmetres: \n", cerca_alpha_lasso.best_params_)


# In[ ]:


coefs_sense_G = cerca_alpha_lasso.best_estimator_.coef_
importancia_sense_G = np.abs(coefs_sense_G)
importancia_sense_G


# In[ ]:


X_sense_G = Dataset_canvis[np.array(X_sense_G.columns)[importancia_sense_G > 0].tolist()]


# In[ ]:


correlacio_X_sense_G = X_sense_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_sense_G, annot=True, linewidths=.5)


# In[ ]:


X_sense_G = X_sense_G.drop(['no_Paid'], axis=1)


# In[ ]:


correlacio_X_sense_G = X_sense_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_sense_G, annot=True, linewidths=.5)


# ### Cerca de millors paràmetres amb G1 i G2:

# In[ ]:


cerca_alpha_lasso.fit(X_amb_G_train_FS, y_amb_G_train_FS.values.ravel())


# In[ ]:


print("Resultats Grid Search: \n")
print("Millor estimador dels paràmetres buscats: \n", cerca_alpha_lasso.best_estimator_)
print("Millor score dels paràmetres: \n", cerca_alpha_lasso.best_score_)
print("Millors paràmetres: \n", cerca_alpha_lasso.best_params_)


# In[ ]:


coefs_amb_G = cerca_alpha_lasso.best_estimator_.coef_
importancia_amb_G = np.abs(coefs_amb_G)
importancia_amb_G


# In[ ]:


X_amb_G = Dataset_canvis[np.array(X_amb_G.columns)[importancia_amb_G > 0].tolist()]


# In[ ]:


correlacio_X_amb_G = X_amb_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_amb_G, annot=True, linewidths=.5)


# In[ ]:


X_amb_G = X_amb_G.drop(['no_Schoolsup', 'no_Paid', 'no_Activities', 'no_Romantic', 'other_Fjob', 'no_Higher','G1'], axis=1)


# In[ ]:


correlacio_X_amb_G = X_amb_G.corr()

plt.figure(figsize = (20,16))

ax = sns.heatmap(correlacio_X_amb_G, annot=True, linewidths=.5)


# ## Regressió RandomForest

# In[ ]:


rf = RandomForestRegressor(random_state=0)


# In[ ]:


grid_param = {
    'n_estimators':np.arange(80,150,10),
    'criterion':['squared_error','absolute_error','poisson'],
    'max_depth': np.arange(2,20,1), 
    'min_samples_leaf': np.arange(1,10,1),
    'min_samples_split': np.arange(2,10,1),
    'max_features':['sqrt', 'log2', 'auto', 1.0]
}


# In[ ]:


cerca_params_rf = GridSearchCV(rf, grid_param, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")


# ### Cerca de millors paràmetres sense G1 i G2:

# In[ ]:


#cerca_params_rf.fit(X_sense_G_train, y_sense_G_train.values.ravel())


# ### Cerca de millors paràmetres amb G1 i G2:

# In[ ]:


#cerca_params_rf.fit(X_amb_G_train_SS, y_amb_G_train.values.ravel())


# # Regressions:

# In[ ]:


Dic_MSE_Sense_G = {}
Dic_MSE_Amb_G = {}

Dic_r2_Sense_G = {}
Dic_r2_Amb_G = {}


# In[ ]:


X_sense_G_train, X_sense_G_test, y_sense_G_train, y_sense_G_test = train_test_split(X_sense_G,y,test_size=0.3,random_state = 23)


# In[ ]:


X_sense_G_train = StandardScaler().fit_transform(X_sense_G_train)
X_sense_G_test = StandardScaler().fit_transform(X_sense_G_test)


# In[ ]:


X_amb_G_train, X_amb_G_test, y_amb_G_train, y_amb_G_test = train_test_split(X_amb_G,y,test_size=0.3,random_state = 50)


# In[ ]:


X_amb_G_train = StandardScaler().fit_transform(X_amb_G_train)
X_amb_G_test = StandardScaler().fit_transform(X_amb_G_test)


# ## Regressió linial sense G1 ni G2:

# In[ ]:


LinReg = LinearRegression()


# In[ ]:


LinReg.fit(X_sense_G_train, y_sense_G_train)


# In[ ]:


Pred_LinReg_sense_G = LinReg.predict(X_sense_G_test)


# ## Regressió linial amb G1 i G2:

# In[ ]:


LinReg.fit(X_amb_G_train, y_amb_G_train)


# In[ ]:


Pred_LinReg_amb_G = LinReg.predict(X_amb_G_test)


# In[ ]:


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

# In[ ]:


Svr = SVR()


# In[ ]:


Svr.fit(X_sense_G_train, y_sense_G_train.values.ravel())


# In[ ]:


Pred_SVR_sense_G = Svr.predict(X_sense_G_test)


# ## SVR amb G1 i G2:

# In[ ]:


Svr.fit(X_amb_G_train, y_amb_G_train.values.ravel())


# In[ ]:


Pred_SVR_amb_G = Svr.predict(X_amb_G_test)


# In[ ]:


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

# In[ ]:


BR = BayesianRidge()


# In[ ]:


BR.fit(X_sense_G_train, y_sense_G_train.values.ravel())


# In[ ]:


Pred_BR_sense_G = BR.predict(X_sense_G_test)


# ## BayesianRidge amb G1 i G2:

# In[ ]:


BR.fit(X_amb_G_train, y_amb_G_train.values.ravel())


# In[ ]:


Pred_BR_amb_G = BR.predict(X_amb_G_test)


# In[ ]:


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

# In[ ]:


CBR = CatBoostRegressor()


# In[ ]:


CBR.fit(X_sense_G_train, y_sense_G_train.values.ravel(), verbose = False)


# In[ ]:


Pred_CBR_sense_G = CBR.predict(X_sense_G_test)


# ## Regressió CatBoost amb G1 i G2:

# In[ ]:


CBR.fit(X_amb_G_train, y_amb_G_train.values.ravel(), verbose = False)


# In[ ]:


Pred_CBR_amb_G = CBR.predict(X_amb_G_test)


# In[ ]:


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

# In[ ]:


XGB = XGBRegressor()


# In[ ]:


XGB.fit(X_sense_G_train, y_sense_G_train)


# In[ ]:


Pred_XGB_sense_G = XGB.predict(X_sense_G_test)


# ## Regressió XGBoost amb G1 i G2:

# In[ ]:


XGB.fit(X_amb_G_train, y_amb_G_train)


# In[ ]:


Pred_XGB_amb_G = XGB.predict(X_amb_G_test)


# In[ ]:


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


# In[ ]:


print("Model amb MSE més baix sense G1 ni G2: ", min(Dic_MSE_Sense_G, key=Dic_MSE_Sense_G.get), "---->",Dic_MSE_Sense_G[min(Dic_MSE_Sense_G, key=Dic_MSE_Sense_G.get)] )
print("Model amb r2 score més alta sense G1 ni G2: ", max(Dic_r2_Sense_G, key=Dic_r2_Sense_G.get), "---->",Dic_r2_Sense_G[max(Dic_r2_Sense_G, key=Dic_r2_Sense_G.get)] )
print("Model amb MSE més baix amb G1 i G2: ", min(Dic_MSE_Amb_G, key=Dic_MSE_Amb_G.get), "---->",Dic_MSE_Amb_G[min(Dic_MSE_Amb_G, key=Dic_MSE_Amb_G.get)])
print("Model amb r2 score més alta amb G1 i G2: ", max(Dic_r2_Amb_G, key=Dic_r2_Amb_G.get), "---->",Dic_r2_Amb_G[max(Dic_r2_Amb_G, key=Dic_r2_Amb_G.get)])


# # Cross-Validation:

# In[ ]:


X_sense_G_cross = Dataset_canvis.copy()
X_sense_G_cross = X_sense_G_cross.drop(['G1','G2','G3'], axis=1)
X_sense_G_cross = StandardScaler().fit_transform(X_sense_G_cross)

X_amb_G_cross = Dataset_canvis.copy()
X_amb_G_cross = X_amb_G_cross.drop(['G3'], axis=1)
X_amb_G_cross = StandardScaler().fit_transform(X_amb_G_cross)

y_cross = Dataset_canvis.copy()
y_cross = y_cross[['G3']]


# ## CV sense G1 ni G2:

# In[ ]:


SS_3 = ShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 0)
SS_5 = ShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 0)
SS_10 = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 0)

KF_3 = KFold(n_splits = 3)
KF_5 = KFold(n_splits = 5)
KF_10 = KFold(n_splits = 10)

loo = LeaveOneOut()


# ### Regressió linial (3, 5 i 10 folds):

# In[ ]:


scores3_LinReg = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=SS_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_LinReg), np.std(scores3_LinReg)))


# In[ ]:


scores5_LinReg = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=SS_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_LinReg), np.std(scores5_LinReg)))


# In[ ]:


scores10_LinReg = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=SS_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_LinReg), np.std(scores10_LinReg)))


# In[ ]:


scores3_LinReg = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=KF_3, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores3_LinReg), np.std(scores3_LinReg)))


# In[ ]:


scores5_LinReg = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=KF_5, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores5_LinReg), np.std(scores5_LinReg)))


# In[ ]:


scores10_LinReg = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=KF_10, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scores10_LinReg), np.std(scores10_LinReg)))


# In[ ]:


scoresloo_LinReg = cross_val_score(LinReg, X_sense_G_cross, y_cross.values.ravel(), cv=loo, scoring='neg_mean_absolute_error')
print('neg_mean_absolute_error mitjà = %.3f amb desviació estàndard = %.3f' % (np.mean(scoresloo_LinReg), np.std(scoresloo_LinReg)))

