import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
VaderSentimentAnalyzer = SentimentIntensityAnalyzer()
# Create a function to compute negative, neutral and positive score
def getAnalysis(score):
  if score <= -0.05:
    return 'Negative'
  elif (score > -0.05) and (score < 0.05):
    return 'Neutral'
  else:
    return 'Positive'

# Label Encoding
def Polarity(score):
  if score == 'Negative':
    return -1
  elif score == 'Neutral':
    return 0
  else:
    return 1

def predict_sentiment_VADER(search_term):
	df1 = pd.read_csv('preprocessed_' + search_term + '_tweets_data.csv', encoding = 'utf8')
	df1 = df1.dropna(subset=['preprocesstweet'])
	df1['scores'] = df1['preprocesstweet'].dropna().apply(lambda Text: VaderSentimentAnalyzer.polarity_scores(Text))
	df1['compound']  = df1['scores'].dropna().apply(lambda score_dict: score_dict['compound'])
	df1 = df1[['tweet','preprocesstweet', 'scores', 'compound']]
	df1['Polarity'] = df1['compound'].apply(getAnalysis)
	df1['Polarity_Score'] = df1['Polarity'].apply(Polarity)
	new_df = df1[['tweet', 'Polarity']]
	new_df.to_csv('Vader_Sentiments.csv', index = False)
