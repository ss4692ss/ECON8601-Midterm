
import pandas

import dill as pickle

import pandas as pd 


with open("text_analysis_machine.pickle", "rb") as f:
  machine = pickle.load(f)
  count_vectorizer_transformer = pickle.load(f)
  lemmatizer = pickle.load(f)
  stopwords = pickle.load(f)
  string = pickle.load(f)
  pre_processing = pickle.load(f)

new_reviews = pd.read_csv("sample_data.csv")


new_reviews_transformed = count_vectorizer_transformer.transform(new_reviews.iloc[:,1])


prediction = machine.predict(new_reviews_transformed)
prediction_prob = machine.predict_proba(new_reviews_transformed)

print(prediction)
print(prediction_prob)

new_reviews['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)


prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
  prediction_prob_dataframe.columns[0]: "prediction_prob_1",
  prediction_prob_dataframe.columns[1]: "prediction_prob_2",
  prediction_prob_dataframe.columns[2]: "prediction_prob_3"
  })

new_reviews = pandas.concat([new_reviews,prediction_prob_dataframe], axis=1)

print(new_reviews)


new_reviews = new_reviews.rename(columns={
  new_reviews.columns[0]: "text"
  })

new_reviews['prediction'] = new_reviews['prediction'].astype(int)
new_reviews['prediction_prob_1'] = round(new_reviews['prediction_prob_1'],4)
new_reviews['prediction_prob_2'] = round(new_reviews['prediction_prob_2'],4)
new_reviews['prediction_prob_3'] = round(new_reviews['prediction_prob_3'],4)


new_reviews.to_csv("new_reviews_with_prediction.csv", index=False)