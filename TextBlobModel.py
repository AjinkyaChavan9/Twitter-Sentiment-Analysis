import pandas as pd
from textblob import TextBlob
# Create a function to get the subjectivity
def getSubjectivity(text):
	return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
	return  TextBlob(text).sentiment.polarity

# Create a function to compute negative, neutral and positive score for TextBlob
def getAnalysisTextBlob(score):
	if score < 0:
		return 'Negative'
	elif score == 0:
		return 'Neutral'
	else:
		return 'Positive'

def predict_sentiment_TextBlob(search_term):
	df1 = pd.read_csv('preprocessed_' + search_term + '_tweets_data.csv', error_bad_lines=False, engine='python', encoding = 'utf8')
	# Create two new columns 'Subjectivity' & 'Polarity_Score'
	df1['Subjectivity'] = df1['preprocesstweet'].dropna().apply(getSubjectivity)
	df1['Polarity_Score'] = df1['preprocesstweet'].dropna().apply(getPolarity)
	df1['Polarity'] = df1['Polarity_Score'].apply(getAnalysisTextBlob)
	new_df = df1[['tweet', 'Polarity', 'Subjectivity','Polarity_Score']]
	new_df.to_csv('TextBlob_Sentiments.csv', index = False)
