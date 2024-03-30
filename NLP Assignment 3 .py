#!/usr/bin/env python
# coding: utf-8

# In[6]:





# In[11]:


import pandas as pd

#https://www.kaggle.com/code/thiagopanini/predicting-credit-risk-eda-viz-pipeline/notebook
df = pd.read_csv("D:\\german_credit_data.csv")

print(df.head())


# In[27]:


#initialize an instance of the GaussianMixture class
from sklearn.mixture import GaussianMixture

#inputs = age and spending score
Y = df[['Age', 'Credit amount']].copy()


#considering three clusters and fit the model to inputs (age and spending score):
n_clusters = 4
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(Y)

#cluster lables
cluster_labels = gmm_model.predict(Y)
Y = pd.DataFrame(Y)
Y['cluster'] = cluster_labels


#plot each cluster within a for-loop
for k in range(0,n_clusters):
    data = Y[Y["cluster"]==k]
    plt.scatter(data["Age"],data["Credit amount"])
    
 
#format out plot
plt.title("Clusters Identified by Guassian Mixture Model : Age vs Credit Amount")    
plt.ylabel("Credit amount")
plt.xlabel("Age")
plt.show()


# In[28]:


#initialize an instance of the GaussianMixture class
from sklearn.mixture import GaussianMixture

#inputs = age and spending score
Y = df[['Age', 'Duration']].copy()


#considering three clusters and fit the model to inputs (age and spending score):
n_clusters = 4
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(Y)

#cluster lables
cluster_labels = gmm_model.predict(Y)
Y = pd.DataFrame(Y)
Y['cluster'] = cluster_labels


#plot each cluster within a for-loop
for k in range(0,n_clusters):
    data = Y[Y["cluster"]==k]
    plt.scatter(data["Age"],data["Duration"])
    
 
#format out plot
plt.title("Clusters Identified by Guassian Mixture Model : Age vs Duration")    
plt.ylabel("Duration")
plt.xlabel("Age")
plt.show()


# In[29]:


#initialize an instance of the GaussianMixture class
from sklearn.mixture import GaussianMixture

#inputs = age and spending score
Y = df[['Credit amount', 'Duration']].copy()


#considering three clusters and fit the model to inputs (age and spending score):
n_clusters = 4
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(Y)

#cluster lables
cluster_labels = gmm_model.predict(Y)
Y = pd.DataFrame(Y)
Y['cluster'] = cluster_labels


#plot each cluster within a for-loop
for k in range(0,n_clusters):
    data = Y[Y["cluster"]==k]
    plt.scatter(data["Credit amount"],data["Duration"])
    
 
#format out plot
plt.title("Clusters Identified by Guassian Mixture Model : Credit Amount vs Duration")    
plt.ylabel("Duration")
plt.xlabel("Credit amount")
plt.show()


# In[ ]:


#below code
#https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/


# In[39]:


Z = df.iloc[:, [7, 8]].values

from sklearn.cluster import KMeans
wcss = [] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Z) 
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method : K value vs WCSS")    
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


# In[42]:


#train the model on the input data with a number of clusters 4

kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = 42)
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
            
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')

plt.xlabel('Credit amount') 
plt.ylabel('Duration') 
plt.legend() 

plt.show()
            


# In[ ]:




