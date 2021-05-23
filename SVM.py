import pandas as pd
import pickle
from joblib import dump, load
import sklearn
SVM = load('Models/SVM.joblib') 


def getAnalysisSVM(score):
	if score == 0:
		return 'Negative'
	else:
		return 'Positive'

def predict_sentiment_SVM(search_term):
	df1 = pd.read_csv('preprocessed_' + search_term + '_tweets_data.csv', error_bad_lines=False, engine='python', encoding = 'utf8')
	df1['Polarity_Score'] = SVM.predict(df1['preprocesstweet'])
	df1['Polarity'] = df1['Polarity_Score'].apply(getAnalysisSVM)
	new_df = df1[['tweet', 'Polarity']]
	new_df.to_csv('SVM_Sentiments.csv', index = False)
