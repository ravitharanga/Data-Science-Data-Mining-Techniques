#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
from spacy import displacy

spacy.cli.download("en_core_web_sm")



# In[2]:


NER = spacy.load("en_core_web_sm")

review_text="I refused a living room set that they sent one week ahead of schedule. I refused it while there was "
"still one week left to the delivery date. They brought it one week ahead of schedule and then charged "
"me a $155 'shipping return and restocking fee' saying it was the "

text1= NER(review_text)

for word in text1.ents:
    print(word.text,word.label_)
    
    
    


# In[3]:


text_from_bbc = "The US is conducting unarmed UAV flights over Gaza, as well as providing advice and assistance to support our Israeli partner as they work on their hostage recovery efforts, the Pentagon's statement on Friday said.The confirmation comes after reporters spotted MQ-9 Reapers, usually operated by American special forces, circling Gaza on Flightradar24, a publicly available flight-tracking website."

text2= NER(text_from_bbc)
    
displacy.render(text2,style="ent",jupyter=True)    
    


# In[4]:


spacy.explain("GPE")


# In[8]:


import pandas as pd
df = pd.read_csv('D:\Walmart_reviews_data.csv')
df


# In[9]:


df['Review'] = df['Review'].str.lower()
df['Review']


# In[12]:


reviews= NER(' '.join(df['Review']))

for word in reviews.ents:
    print(word.text,word.label_)



# In[14]:


displacy.render(NER(' '.join(df['Review'])),style="ent",jupyter=True)    


# In[18]:


import nltk
from nltk import ne_chunk #importing chunk library from nltk
from nltk.tokenize import word_tokenize

text_from_bbc_2 = "The US is conducting unarmed UAV flights over Gaza, as well as providing advice and assistance to support our Israeli partner as they work on their hostage recovery efforts, the Pentagon's statement on Friday said.The confirmation comes after reporters spotted MQ-9 Reapers, usually operated by American special forces, circling Gaza on Flightradar24, a publicly available flight-tracking website."

token = word_tokenize(text_from_bbc_2) #tokenizing 

tags = nltk.pos_tag(token) #tagging

chunk = ne_chunk(tags)

chunk


# In[ ]:




