import pandas as pd
import pickle
from joblib import dump, load
import sklearn
LogReg = load('Models/LogReg.joblib') 


def getAnalysisLogReg(score):
	if score == 0:
		return 'Negative'
	else:
		return 'Positive'

def predict_sentiment_LogReg(search_term):
	df1 = pd.read_csv('preprocessed_' + search_term + '_tweets_data.csv', error_bad_lines=False, engine='python', encoding = 'utf8')
	df1['Polarity_Score'] = LogReg.predict(df1['preprocesstweet'])
	df1['Polarity'] = df1['Polarity_Score'].apply(getAnalysisLogReg)
	new_df = df1[['tweet', 'Polarity']]
	new_df.to_csv('Logistic_Regression_Sentiments.csv', index = False)
