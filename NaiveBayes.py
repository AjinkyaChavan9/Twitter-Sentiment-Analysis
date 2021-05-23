import pandas as pd
import pickle
from joblib import dump, load
import sklearn
NB = load('Models/NB.joblib') 


def getAnalysisNB(score):
	if score == 0:
		return 'Negative'
	else:
		return 'Positive'

def predict_sentiment_NB(search_term):
	df1 = pd.read_csv('preprocessed_' + search_term + '_tweets_data.csv', error_bad_lines=False, engine='python', encoding = 'utf8')
	df1['Polarity_Score'] = NB.predict(df1['preprocesstweet'])
	df1['Polarity'] = df1['Polarity_Score'].apply(getAnalysisNB)
	new_df = df1[['tweet', 'Polarity']]
	new_df.to_csv('NB_Sentiments.csv', index = False)
