#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd


# In[84]:


df = pd.read_csv('D:\Walmart_reviews_data.csv')


# In[85]:


print(df.shape)


# In[86]:


print(df.dtypes)


# In[87]:


print(df.isnull().sum())


# In[88]:


df.drop(['Image_Links'], axis = 1, inplace = True)


# In[89]:


print(df.shape)


# In[90]:


print(df.info)


# In[91]:


print(df.tail())


# In[92]:


df['Review'] = df['Review'].str.lower()


# In[93]:


print(df.tail())


# In[94]:


import string
df['Review'] = df['Review'].str.translate(str.maketrans('','', string.punctuation))


# In[95]:


print(df.tail())


# In[96]:


print(len(df['Review']))


# In[97]:


import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
print(stopwords.words('english'))




# In[98]:


nltk.download('stopwords')
stopw_nltk = stopwords.words('english')
print(len(stopw_nltk))


# In[99]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
example_sent = """This is a sample sentence, showing off the stop words filtration."""
 
stop_words = set(stopwords.words('english'))
 
word_tokens = word_tokenize(example_sent)
# converts the words in word_tokens to lower case and then checks whether 
#they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#with no lower case conversion
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
 
print(word_tokens)
print(filtered_sentence)


# In[100]:


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
 
example_sent = str(df['Review']) 
 
stop_words = set(stopwords.words('english'))
 
sent_tokens = sent_tokenize(example_sent)

filtered_paras = [w for w in sent_tokens if not w.lower() in stop_words]

filtered_paras = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_paras.append(w)
 
print(sent_tokens)
print("")
print(filtered_paras)


# In[101]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
example_sent = str(df['Review']) 
 
stop_words = set(stopwords.words('english'))
 
word_tokens = word_tokenize(example_sent)
# converts the words in word_tokens to lower case and then checks whether 
#they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#with no lower case conversion
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
 
print(word_tokens)
print("")
print(filtered_sentence)


# In[102]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[103]:


print (df['Review'])


# In[104]:


print(len(df['Review']))


# In[105]:


pip install spacy


# In[106]:


import spacy.cli

spacy.cli.download("en_core_web_lg")


# In[107]:


import spacy

en = spacy.load('en_core_web_lg')
stopw_spacy = en.Defaults.stop_words

print(stopw_spacy)


# In[108]:


print(len(stopw_spacy))


# In[109]:


df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopw_spacy)]))


# In[110]:


print (df['Review'])


# In[111]:


import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
print(STOPWORDS)


# In[112]:


print(len(STOPWORDS))


# In[113]:


new_review = remove_stopwords(str(df['Review']))
print(new_review)


# In[114]:


from nltk.stem import PorterStemmer

word_Pstemmer = PorterStemmer() 

# Split the sentences to lists of words.
df['Review'] = df['Review'].str.split()

df['stemmed_Review'] = df['Review'].apply(lambda x: [word_Pstemmer.stem(y) for y in x]) # Stem every word.

df 


# In[ ]:




