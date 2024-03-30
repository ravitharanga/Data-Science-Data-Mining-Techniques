#!/usr/bin/env python
# coding: utf-8

# In[5]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))
 
feedback = "The customer service is very bad. I bought a Breville Barista Pro espresso machine for 468$ "
"which is too cheap because the machine's price is 850$. The third-party seller was a scammer." 
    

tokenized = sent_tokenize(feedback)
for i in tokenized:
     
    # Word tokenizers is used to find the words, punctuation
    wordsList = nltk.word_tokenize(i)
 
    # removing stop words 
    wordsList = [w for w in wordsList if not w in stop_words] 
 
    #  Using a Tagger. Which is part-of-speech tagger or POS-tagger
    tagged = nltk.pos_tag(wordsList)
 
    print(tagged)


# In[6]:


import pandas as pd
df = pd.read_csv('D:\Walmart_reviews_data.csv')
df


# In[7]:


df['Review'] = df['Review'].str.lower()
df['Review']


# In[11]:


all_feedbacks_together_reviews = ' '.join(df['Review'])
print(all_feedbacks_together_reviews)


    


# In[12]:


tokenized = sent_tokenize(all_feedbacks_together_reviews)
for i in tokenized:
     
    # Word tokenizers is used to find the words, punctuation
    wordsList = nltk.word_tokenize(i)
 
    # removing stop words 
    wordsList = [w for w in wordsList if not w in stop_words] 
 
    #  Using a Tagger. Which is part-of-speech tagger or POS-tagger
    tagged = nltk.pos_tag(wordsList)
 
    print(tagged)


# In[ ]:




