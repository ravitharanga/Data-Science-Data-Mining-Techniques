#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('D:\Walmart_reviews_data.csv')
df


# In[2]:


df['Review'] = df['Review'].str.lower()
df['Review']


# In[3]:


import string
df['Review'] = df['Review'].str.translate(str.maketrans('','', string.punctuation))
df['Review']


# In[5]:


all_feedbacks_together = ' '.join(df['Review'])
print(all_feedbacks_together)

# Split the sentences to lists of words.
all_feedbacks_together_to_words = all_feedbacks_together.split()
print(all_feedbacks_together_to_words)


# In[6]:


from nltk.stem import LancasterStemmer
l_stammer = LancasterStemmer()

new_list = []

for word in all_feedbacks_together_to_words:
    new_list.append(word)

    result = word
   
    print(result)

print(l_stammer.stem(new_list)) 


# In[9]:


from nltk.stem import LancasterStemmer
l_stammer = LancasterStemmer()

words = ['meeting','meets','met','pushes','pushing']
for word in words:
    print(word,"--->",l_stammer.stem(word))


# In[11]:


from nltk.stem import RegexpStemmer

re_stemmer = RegexpStemmer('ing')

words = ['meeting','meets','catching','pushes','pushing','hiding']
for word in words:
    print(word,"--->",re_stemmer.stem(word))


# In[19]:


re_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
words = ['mass','was','bee','computer','advisable','eating','shouting','streaming','notifies']
for word in words:
    print(word,"--->",re_stemmer.stem(word))


# In[13]:


from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language='english')
words = ['generous','generate','generously','generation','authentication','stimulation','precisely','validate',
         'validation']
for word in words:
    print(word,"--->",snowball.stem(word))


# In[32]:


from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, RegexpStemmer

porter_stemmer = PorterStemmer()

lancaster_stemmer = LancasterStemmer()

snowball_stemmer = SnowballStemmer(language='english')

regexp_stemmer = RegexpStemmer('ing$|s$|able$|b$|ship$', min=6)

word_list = ["horrible", "friendship", "superb", "disgusting"]

print("{0:20}{1:20}{2:20}{3:30}{4:40}".format("Word","Porter Stemmer","Snowball Stemmer","Lancaster Stemmer",
                                              'Regexp Stemmer'))

for word in word_list:
    print("{0:20}{1:20}{2:20}{3:30}{4:40}".format(word,porter_stemmer.stem(word),
                                                  snowball_stemmer.stem(word),
                                                  lancaster_stemmer.stem(word),
                                                  regexp_stemmer.stem(word)))
    
    


# In[ ]:




