#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd

#reading the excel data source
reviews_df = pd.read_excel("D:\Sample_Data_set_for_Task_3.xlsx")


# In[13]:


reviews_df.head()


# In[14]:


reviews_df.tail()


# In[15]:


#Viewing Review Text column
print(reviews_df["Review Text"])


# In[16]:


# create the label
reviews_df["is_bad_review"] = reviews_df["Rating"].apply(lambda x: 1 if x < 5 else 0)
#print(reviews_df["is_bad_review"])


#select only relevant columns
reviews_df = reviews_df[["Review Text", "is_bad_review"]]
reviews_df.head()


# In[17]:


#this is to speed up computations - sample data
reviews_df = reviews_df.sample(frac = 0.1, replace = False, random_state=42)


# In[18]:


#eliminate 'No Negative' or 'No Positive' from text
#need to remove those parts from our texts - data cleaning
reviews_df["Review Text"] = reviews_df["Review Text"].apply(lambda x: x.replace("No Negative", "").
                                                            replace("No Positive", ""))




# In[19]:


# based on the POS rags, returns the wordnet object value
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    




# In[24]:


import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

#function - cleaning review text 
def clean_text(text):
    # lower text - simple letters
    text = text.lower()
    
    # tokenize text (splitting text into words) and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    
    # remove stop words - unnecessary words remover
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    
    # pos tag text - assign a tag to every word to define if it corresponds to a noun, a verb etc.
    pos_tags = pos_tag(text)
    
    # lemmatize text - transform every word into their root form
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    
    # join all
    text = " ".join(text)
    return(text)

# clean Review Text data
reviews_df["review_clean"] = reviews_df["Review Text"].apply(lambda x: clean_text(x))

#printing on screen after cleaning Review text
print(reviews_df["review_clean"])


# In[ ]:


#adding sentiment anaylsis columns - Feature engineering (#neutrality score, positivity score, negativity score, an overall score that summarizes the previous scores)
#integrate those 4 values as features in our dataset.

#Vader - a part of the NLTK module - for sentiment analysis
#Vader sorts words into positive and negative categories using a lexicon. 
#In order to calculate the sentiment scores, it additionally considers the statements' context.


# In[25]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
reviews_df["sentiments"] = reviews_df["Review Text"].apply(lambda x: sid.polarity_scores(x))
reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)

print(reviews_df)


# In[26]:


#adding no of characters column
reviews_df["nb_chars"] = reviews_df["Review Text"].apply(lambda x: len(x))

#adding no of words column
reviews_df["nb_words"] = reviews_df["Review Text"].apply(lambda x: len(x.split(" ")))



# In[ ]:


#add some simple metrics for every text:

#number of characters in the text
#number of words in the text


# In[27]:


# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["review_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with the Review text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)

print(reviews_df)


# In[ ]:





# In[ ]:


#extracting vector representations for every review

#Using the context in which the words appear, the Gensim module generates a numerical vector representation 
#of each word in the corpus (Word2Vec).

#Word vectors (Doc2Vec) can also be used to convert any text into numerical vectors. 
#Since related texts will also have similar representations, those vectors can be used as training features.

#first have to train a Doc2Vec model by feeding in the Review text data


# In[53]:


# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)


#print(reviews_df)


# In[32]:


pip install wordcloud


# In[33]:


# wordcloud function

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(reviews_df["Review Text"])


# In[ ]:


#most of the words are positive, means customers are happy about the products 


# In[48]:


#printing "nb_words" column
print(reviews_df["nb_words"]) 


# In[52]:


print(reviews_df) 


# In[49]:


# highest positive sentiment reviews (with more than 5 words)
reviews_df[reviews_df["nb_words"] >= 5].sort_values("nb_words", ascending = False)[["Review Text", "nb_words"]].head(10)


# In[50]:


# lowest negative sentiment reviews (with more than 5 words)
reviews_df[reviews_df["nb_words"] >= 5].sort_values("neg", ascending = False)[["Review Text", "neg"]].head(10)



# In[42]:


# generating a plot to view positive and negative review feedbacks

import seaborn as sns

for x in [0, 1]:
    subset = reviews_df[reviews_df['is_bad_review'] == x]
    
    # Draw the density plot
    if x == 0:
        label = "Good reviews"
    else:
        label = "Bad reviews"
    sns.distplot(subset['compound'], hist = False, label = label)
    


# In[ ]:


#The sentiment distribution between positive and negative reviews is displayed.
#The blue line indicates positive reviews and the orange line indicates negative reviews

 
#It is evident that Vader regards positive reviews for the majority of them as extremely positive. 
#Bad reviews, on the other hand, typically have lower compound sentiment scores.


# In[43]:


#selecting features

#one set of features to train our model and the other to assess its performances

label = "is_bad_review"
ignore_cols = [label, "Review Text", "review_clean"]
features = [c for c in reviews_df.columns if c not in ignore_cols]

# split the data into train and test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(reviews_df[features], reviews_df[label], test_size = 0.20, random_state = 42)



# In[44]:


# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(20)




# In[45]:


# ROC curve

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

y_pred = [x[1] for x in rf.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




# In[46]:


# PR curve

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.utils.fixes import signature

average_precision = average_precision_score(y_test, y_pred)

precision, recall, _ = precision_recall_curve(y_test, y_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

plt.figure(1, figsize = (15, 10))
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))




# In[ ]:




