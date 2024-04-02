import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import string

import pandas as pd

import pandas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import dill as pickle




review = pd.read_csv("dataset.csv")

data = review["reviewtext"]
target = review["stars"]


lemmatizer = WordNetLemmatizer()


def pre_processing(reviewtext):

    text_processed = reviewtext.translate(str.maketrans("", "", string.punctuation))

    text_processed = text_processed.split()
    result = []
    for word in text_processed:
        word_processed = word.lower()
        if word_processed not in stopwords.words("english"):
            word_processed = lemmatizer.lemmatize(word_processed)
            result.append(word_processed)
    return " ".join(result)


count_vectorizer_transformer = CountVectorizer(analyzer=pre_processing).fit(data)


data_transformed = count_vectorizer_transformer.transform(data)


machine = MultinomialNB()
machine.fit(data_transformed, target)


with open("text_analysis_machine.pickle", "wb") as f:
  pickle.dump(machine, f)
  pickle.dump(count_vectorizer_transformer, f)
  pickle.dump(lemmatizer, f)
  pickle.dump(stopwords, f)
  pickle.dump(string, f)
  pickle.dump(pre_processing, f)





