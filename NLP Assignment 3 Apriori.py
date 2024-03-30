#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#No header so header = None
dataset = pd.read_csv('D:\Market_Basket_Optimisation.csv', header = None) 
dataset.shape



# In[2]:


#Generate a list of lists = this is to index transactions 
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

print(transactions[0])


# In[3]:


from apyori import apriori

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

results = list(rules)


# In[4]:


#generate a table
results = pd.DataFrame(results)
results.head(10)


# In[5]:


dataset.head()


# In[6]:


dataset.tail()


# In[7]:


#convert pandas dataframe into a list of lists
records = []
for i in range(0, 7501):
    records.append([str(dataset.values[i,j]) for j in range(0, 20)])


# In[8]:


#Applying Apriori

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)


#total no of rules

print(len(association_results))


# In[9]:


#get the 1st rule

print(association_results[0])


# In[10]:


#we can see that light cream and chicken are commonly bought together

#The support value for the first rule is 0.0045. 
#This number is calculated by dividing the number of transactions containing light cream 
#divided by total number of transactions


#The confidence level for the rule is 0.2905 which shows that out of all the transactions that contain light cream, 29.05% 
#of the transactions also contain chicken.

#the lift of 4.84 tells us that chicken is 4.84 times more likely to be bought by the customers who buy 
#light cream compared to the default likelihood of the sale of chicken.






# In[11]:


for item in association_results:

    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1] + " "+ "Support: " + str(item[1]) +" "+ "Confidence: " + str(item[2][0][2]) +" "+ "Lift: " + str(item[2][0][3]))

  



# In[12]:


#generate a table
results = pd.DataFrame(association_results)
results.head(10)


# In[13]:


#above
#https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/

