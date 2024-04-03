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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import kfold_template
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics


review = pd.read_csv("/Users/sanjeevsharma/Desktop/Midterm861/dataset.csv")
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


##Multinomia

X_train, X_test, y_train, y_test = train_test_split(data_transformed, review["stars"], test_size=0.2, random_state=42)
machine.fit(X_train, y_train)
predicted_labels = machine.predict(X_test)
accuracy = accuracy_score(y_test, predicted_labels)
print("Accuracy Score:", accuracy)

##Random Forest

machine = RandomForestClassifier(criterion="gini", max_depth=2, n_estimators=100, bootstrap = True) 
return_values = kfold_template.run_kfold(machine, data_transformed, review["stars"], 4, False)
print(return_values)


##logistic 

X_train, X_test, y_train, y_test = train_test_split(data_transformed, review["stars"], test_size=0.2, random_state=42)
machine = linear_model.LogisticRegression()
machine.fit(X_train, y_train)
prediction = machine.predict(X_test)
accuracy_score = metrics.accuracy_score(y_test, prediction)
print("Accuracy score: ", accuracy_score)
  


