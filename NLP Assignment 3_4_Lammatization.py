#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
df = pd.read_csv('D:\Walmart_reviews_data.csv')
df


# In[8]:


df['Review'] = df['Review'].str.lower()
df['Review']


# In[9]:


import string
df['Review'] = df['Review'].str.translate(str.maketrans('','', string.punctuation))
df['Review']


# In[10]:


all_feedbacks_together = ' '.join(df['Review'])
print(all_feedbacks_together)

# Split the sentences to lists of words.
all_feedbacks_together_to_words = all_feedbacks_together.split()
print(all_feedbacks_together_to_words)


# In[11]:


import nltk 
from nltk.stem import WordNetLemmatizer 

# Define a text string 
sample_text = "words books eating" 

# Tokenizing / individual words 
tokens = nltk.word_tokenize(sample_text) 

# WordNetLemmatizer object / instance 
WN_lemmatizer = WordNetLemmatizer() 

# Lemmatizing  
for token in tokens: 
  la = WN_lemmatizer.lemmatize(token) 
  print(token, "-->", la)


# In[12]:


all_feedbacks_together = ' '.join(df['Review'])
tokens_2 = nltk.word_tokenize(all_feedbacks_together) 

WN_lemmatizer_2 = WordNetLemmatizer() 

for token in tokens_2: 
  la_2 = WN_lemmatizer_2.lemmatize(token) 
  print(token, "-->", la_2)


# In[13]:


pip install spacy


# In[17]:


import spacy.cli

spacy.cli.download("en_core_web_lg")


# In[26]:


import spacy

# Load the English language model in spaCy 
nlp_spacy = spacy.load('en_core_web_lg')

# Define a text string 
text_sample = "This is a sample text and this contains some words" 

# Create a Doc object 
doc = nlp_spacy(text_sample) 

# Lemmatize each token and print the result 
for token in doc: 
  le = token.lemma_ 
  print(token.text, "-->", le)


# In[27]:


nlp_spacy = spacy.load('en_core_web_lg')

text_2 = ' '.join(df['Review'])

doc = nlp_spacy(text_2) 

for token in doc: 
  le = token.lemma_ 
  print(token.text, "-->", le)    


# In[1]:


import gensim 
from gensim.utils import lemmatize 

# Define a text string 
text = "This is a sample text. It contains some words that we can use for lemmatization." 

# Use the lemmatize() function to lemmatize the text 
lemmas = lemmatize(text, stopwords=['is', 'it', 'we']) 

# Print the result 
print(lemmas)


# In[2]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""
               
               
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)  
    print(sentences[i])


# In[ ]:




