#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.kaggle.com/datasets/sindraanthony9985/marketing-data-for-a-supermarket-in-united-states/data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('D:\Market_Basket_Optimisation.csv') 
#dataset.shape


# In[2]:


dataset.head()


# In[3]:


dataset.tail()


# In[2]:


#convert pandas dataframe into a list of lists
records = []
for i in range(0, 7499):
    records.append([str(dataset.values[i,j]) for j in range(0, 3)])


# In[3]:


print(records[0])


# In[4]:


#generate a table
results = pd.DataFrame(records)
results.head(10)


# In[5]:


# Convert the data to a one-hot encoded forma
one_hot_data = pd.get_dummies(results.unstack().dropna()).groupby(level=1).sum()


# In[9]:


from apyori import apriori

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(one_hot_data, min_support=0.0045, use_colnames=True)


# In[16]:


conda install tensorflow


# In[1]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from tensorflow.python.keras.utils.data_utils import Sequence





# In[25]:


#Applying Apriori
from apyori import apriori

#min_length=2 minimum 2 items for each row

association_rules_1 = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=2, min_length=2)
association_results = list(association_rules_1)


#total no of rules

print(len(association_results))


# In[26]:


print(association_results[0])


# In[9]:


for item in association_results:

    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1] + " "+ "Support: " + str(item[1]) +" "+ "Confidence: " + str(item[2][0][2]) +" "+ "Lift: " + str(item[2][0][3]))


# In[27]:


for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


# In[11]:


#generate a table
results = pd.DataFrame(association_results)
results.head(10)


# In[12]:


pip install mlxtend


# In[20]:


#Apriori Algorithm and One-Hot Encoding

#Apriori's algorithm transforms True/False or 1/0.
#Using TransactionEncoder, we convert the list to a One-Hot Encoded Boolean list.
#Products that customers bought or did not buy during shopping will now be represented by values 1 and 0.


#Let's transform the list, with one-hot encoding

from mlxtend.preprocessing import TransactionEncoder

a = TransactionEncoder()
a_data = a.fit(dataset).transform(dataset)
df = pd.DataFrame(a_data,columns=a.columns_)
df = df.replace(False,0)
df


# In[22]:


# One-hot encoding
te = TransactionEncoder()
data_transformed = te.fit_transform(dataset)
df = pd.DataFrame(data_transformed, columns=te.columns_)
df


# In[28]:


# Find frequent item sets
frequent_items = apriori(df, min_support = 0.6, use_colnames = True)

# Generate strong association rules
rules = association_rules(frequent_items, metric ="confidence", min_threshold = 0.8)
rules = rules.sort_values(by='confidence', ascending =False)
rules
 


# In[ ]:





# In[ ]:





# In[ ]:




