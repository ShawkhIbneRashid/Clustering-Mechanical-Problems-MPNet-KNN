from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import Word
import pandas as pd
import numpy as np
import re
import string

#clas to load the dataset and perform some data preprocessing. This class also outputs the number of unique uids in the dataset
class Load_Preprocess():

  def __init__(self,directory):
    self.directory = directory

  def load_data(self):
    df = pd.read_csv(self.directory)
    df.rename(columns={'Extracted problems':'extracted_problems'},inplace=True)
    return df

  def preprocess(self, df):
    #making all the letters lowercase
    df['extracted_problems'] = [text.lower() for text in df['extracted_problems']]

    #removing punctuations
    df['extracted_problems'] = df['extracted_problems'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))

    #removing stop words
    stop = stopwords.words('english')
    df['extracted_problems'] = df['extracted_problems'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    #performing lemmatization to change the words in their base forms
    df['extracted_problems'] = df['extracted_problems'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    #removing extra spaces
    df["extracted_problems"] = df["extracted_problems"].apply(lambda text: re.sub(' +', ' ', text))
    return df

  def unique_uid(self, df):
    #finding the number of unique uid in the dataset
    #all the uids are different from each other
    #so I have considered the first 2 letters of all the uids
    #as there are total 6 unique first two letters of the uid, considered 6 unique uid
    #so all the records now belong to these 6 categories or labels or uids
    #the next step is to cluster the respective records of these 6 labels
    uid_2 = []
    uid_2 = [i[0:2] for i in df['uid']]
    uid_set = set(uid_2)
    unique_uid = len(uid_set)
    print('Unique uids are:')
    print(uid_set)
    print('Total number of unique uid: ' + str(unique_uid))
