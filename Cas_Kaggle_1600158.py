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

