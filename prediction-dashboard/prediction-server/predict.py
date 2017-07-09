
# coding: utf-8

# # EDA/Pre-processing
# Initial EDA. 
# Takes in the initial text data and returns various pre-processed data sets to the data folder.

# ## STEP 1: File Input

# In[3]:

import pandas as pd
import os
import numpy as np

# NLP
import nltk


# In[4]:
txt_file = 'yelp_labelled.txt';
df = pd.read_table(txt_file, sep='\t')
df.columns = ['reviews','sentiment']


# ## STEP 3: Clean data

# ### Count_unique method
# Allows us to count the number of unique words which will be the number of words in our bag-of-words model. A simple statistic for basic insight on the impact of our data cleaning.

# In[5]:

def count_unique(words):
    uniq = set()
    for sentence in words:
        for word in sentence:
            uniq.add(word)
    return len(uniq)


# ### Tokenize

# In[6]:

#Remove problems with encoding
df['reviews'] = df['reviews'].apply(lambda x: unicode(x, errors="ignore"))


# In[7]:

#tokenize without the punctuation
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
# make reviews lowercase
df['reviews'] = df['reviews'].apply(lambda x: x.lower())
# tokenize
df['reviews'] = df['reviews'].apply(lambda x: tokenizer.tokenize(x))


# ### Remove Stopwords

# In[8]:

# Import the list of stopwords from NLTK
from nltk.corpus import stopwords

# Remove the stopwords and store in "No Stops" column
stop = set(stopwords.words('english'))
# Ensure "not" is kept
stop.remove("not")
df['stpd'] = df['reviews'].apply(lambda x: [item for item in x if item not in stop])
df['nstpd'] = df['reviews']


# ### POS Tagging/Removal

# In[9]:

# POS tag
df['stpd_posr']= df['stpd'].apply(lambda x: nltk.pos_tag(x))
df['nstpd_posr']= df['nstpd'].apply(lambda x: nltk.pos_tag(x))
# Leave the non POS removal ones alone
df['stpd_nposr'] = df['stpd']
df['nstpd_nposr'] = df['nstpd']


# In[10]:

# Collection of adjectives, nouns, adverbs and verbs to keep
pos_keep = ["JJ","JJR","JJS","NN","NNP","NNS","RB","RBR","VB","VBD","VBG","VBN","VBZ"]

def remove_pos(full):
    redc =[]
    for pair in full:
        if pair[1] in pos_keep:
            redc.append(pair[0])
    return redc
# Remove the words that are not in pos_keep
df['stpd_posr']= df['stpd_posr'].apply(lambda x: remove_pos(x))
df['nstpd_posr']= df['nstpd_posr'].apply(lambda x: remove_pos(x))


# ### Stemming.
# We will test three different stemmers. Porter, Snowball, Lancaster

# In[11]:

ps = nltk.PorterStemmer()
ss = nltk.SnowballStemmer('english')
ls = nltk.LancasterStemmer()

# No Stemming
df['nstpd_nposr_nstem'] = df['nstpd_nposr']
df['nstpd_posr_nstem'] = df['nstpd_posr']
df['stpd_nposr_nstem'] = df['stpd_nposr']
df['stpd_posr_nstem'] = df['stpd_posr']
# Porter
df['nstpd_nposr_port'] = df['nstpd_nposr'].apply(lambda x: [ps.stem(y) for y in x])
df['nstpd_posr_port'] = df['nstpd_posr'].apply(lambda x: [ps.stem(y) for y in x])
df['stpd_nposr_port'] = df['stpd_nposr'].apply(lambda x: [ps.stem(y) for y in x])
df['stpd_posr_port'] = df['stpd_posr'].apply(lambda x: [ps.stem(y) for y in x])
# Snowball
df['nstpd_nposr_snow'] = df['nstpd_nposr'].apply(lambda x: [ss.stem(y) for y in x])
df['nstpd_posr_snow'] = df['nstpd_posr'].apply(lambda x: [ss.stem(y) for y in x])
df['stpd_nposr_snow'] = df['stpd_nposr'].apply(lambda x: [ss.stem(y) for y in x])
df['stpd_posr_snow'] = df['stpd_posr'].apply(lambda x: [ss.stem(y) for y in x])
# Lancaster
df['nstpd_nposr_lanc'] = df['nstpd_nposr'].apply(lambda x: [ls.stem(y) for y in x])
df['nstpd_posr_lanc'] = df['nstpd_posr'].apply(lambda x: [ls.stem(y) for y in x])
df['stpd_nposr_lanc'] = df['stpd_nposr'].apply(lambda x: [ls.stem(y) for y in x])
df['stpd_posr_lanc'] = df['stpd_posr'].apply(lambda x: [ls.stem(y) for y in x])


# ### Output to a file

# In[12]:

# put final to csv just in case need a file

# ## Step 4 : Test and Evaluate with Logistic Regression, Naive Bayes, k-fold

# ### Import Essential Libraries


# ### Import sklearn and NLTK Library and Modules

# In[14]:

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
logreg = SklearnClassifier(LogisticRegression())


# ### Function to Convert the Data into a Feature Set

# In[15]:

## Transform data into list of ([tokens],sentiment label)
def createTrainingDataNLTK(sentences,labels):
    rdata = np.vstack([sentences,labels])
    rdata = np.transpose(rdata)
    data = list();
    for i in range(0,len(rdata)):
        tokens = rdata[i][0].split(" ")
        d_tuple = (tokens, rdata[i][1]);
        data.append(d_tuple)
    return data;


# ## Create the Training Data

# In[16]:

# merge the words into sentence to use current implementation of createTrainingData
def create_nltk_train_data (feature_reduction):
    df['sentences'] = df[feature_reduction].apply(lambda x: " ".join(x).encode('UTF-8'))
    x_label = "sentences"
    y_label = "sentiment"
    nltk_train_data = createTrainingDataNLTK(df[x_label],df[y_label])
    return nltk_train_data


# ## Functions to Train the Classifers and Run it Against Test Data

# In[17]:

# returns the accuracy of the test data for Naive Bayes
# when predicted against fitted training data
def train_nb(training_set):
    
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_set])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    
    training_set = sentim_analyzer.apply_features(training_set)
                                              
    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)                         
    return [sentim_analyzer,classifier]


# In[18]:

# returns the accuracy of the test data for Logistic Regression
# when predicted against fitted training data
def train_lr(training_set):
    
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_set])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    
    training_set = sentim_analyzer.apply_features(training_set)
                                              
    trainer = logreg.train
    classifier = sentim_analyzer.train(trainer, training_set)                 
    return [sentim_analyzer,classifier]


# In[19]:
def predict_lr(text, trained):
    tok =  tokenizer.tokenize(text)
    tok = nltk.pos_tag(tok)
    tok = remove_pos(tok)
    fs = trained[0].apply_features([(tok)])
    return trained[1].prob_classify(fs[0][0])

# In[20]:
def predict_nb(text, trained):
    tok =  tokenizer.tokenize(text)
    tok = nltk.pos_tag(tok)
    tok = remove_pos(tok)
    fs = trained[0].apply_features([(tok)])
    return trained[1].prob_classify(fs[0][0])

# Create the Flask Restful API endpoints to expose the trained models

from flask import Flask
from flask import request
app = Flask(__name__)

ds_l = train_lr(create_nltk_train_data('nstpd_nposr_snow'));
ds_n = train_nb(create_nltk_train_data('nstpd_nposr_nstem'));

@app.route('/predict/algo1/<sentence>')
def predict1(sentence):
    print sentence
    dist = predict_lr(sentence,ds_l)
    score = str(1*dist.prob(1))
    return score

@app.route('/predict/algo2/<sentence>')
def predict2(sentence):
    dist = predict_nb(sentence,ds_n)
    score = str(1*dist.prob(1))
    return score

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


