#!/usr/bin/env python
# coding: utf-8

# In[148]:


# supress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing all required packages
import numpy as np
import pandas as pd

# Data viz lib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import xticks


# In[149]:


bank = pd.read_csv('D://bankmarketing.csv')


# In[150]:


bank.head()


# In[151]:


bank.columns


# In[152]:


# Importing Categorical Columns

bank_cust = bank[['age','job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','day_of_week','poutcome']]




# In[153]:


bank_cust.head()


# In[154]:


# Converting age into categorical variable

bank_cust['age_category'] = pd.cut(bank_cust['age'], [0, 30, 40, 50, 60, 70, 80, 90, 100], 
                              labels=['0-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
bank_cust  = bank_cust.drop('age',axis = 1)

bank_cust.head()


# In[155]:


bank_cust.shape


# In[156]:


bank_cust.describe()


# In[157]:


bank_cust.info()


# In[158]:


# Checking Null values
bank_cust.isnull().sum()*100/bank_cust.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# In[159]:


# First we will keep a copy of data
bank_cust_copy = bank_cust.copy()


# In[160]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bank_cust = bank_cust.apply(le.fit_transform)
bank_cust.head()


# In[161]:


pip install kmodes


# In[162]:


# Importing Libraries

from kmodes.kmodes import KModes


# In[163]:


km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)


# In[164]:


# Predicted Clusters
fitClusters_cao


# In[165]:


clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = bank_cust.columns

# Mode of the clusters
clusterCentroidsDf


# In[166]:


import pandas as pd

df = pd.read_csv("D:\\Mall_Customers.csv")

print(df.head())


# In[170]:


#initialize an instance of the GaussianMixture class
from sklearn.mixture import GaussianMixture

#inputs = age and spending score
X = df[['Age', 'Spending Score (1-100)']].copy()


#considering three clusters and fit the model to inputs (age and spending score):
n_clusters = 4
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(X)

#cluster lables
cluster_labels = gmm_model.predict(X)
X = pd.DataFrame(X)
X['cluster'] = cluster_labels


#plot each cluster within a for-loop
for k in range(0,n_clusters):
    data = X[X["cluster"]==k]
    plt.scatter(data["Age"],data["Spending Score (1-100)"])
    
 
#format out plot
plt.title("Clusters Identified by Guassian Mixture Model")    
plt.ylabel("Spending Score (1-100)")
plt.xlabel("Age")
plt.show()


    


# In[ ]:




