import preprocessor as preprocess
import wordtodigits
import pandas as pd
import nltk
import re


def preprocess_function(search_term):
	df = pd.read_csv(search_term + '_tweets_data.csv')
	# defining preprocess function for each row
	def preprocess_tweet(row):
		text = row['tweet']
		text = preprocess.clean(text)
		#Replace Clitics with their full forms
		text = re.sub(r"what's", "what is ", text)
		text = re.sub(r"\'s", " ", text) #remove apostrophe 's
		text = re.sub(r"\'ve", " have ", text)
		text = re.sub(r"can't", "cannot ", text)
		text = re.sub(r"n't", " not ", text)
		text = re.sub(r"i'm", "i am ", text)
		text = re.sub(r"\'re", " are ", text)
		text = re.sub(r"\'d", " would ", text)
		text = re.sub(r"\'ll", " will ", text)
		text = re.sub(r"\'scuse", " excuse ", text)
		return text

	df['preprocesstweet'] = df.apply(preprocess_tweet, axis=1)

	#Lower Casing
	df['preprocesstweet'] = df['preprocesstweet'].str.lower()

	#Convert word numbers to digit
	df['preprocesstweet'] = df['preprocesstweet'].apply(lambda row: wordtodigits.convert(row))
	#removing digit numbers
	df['preprocesstweet'] = df['preprocesstweet'].str.replace(r'\d+','', regex=True)

	#Removing punctuation marks with python RegEx(Regular Expression) 
	# [] - A set of characters
	# \w - Returns a match where the string contains any word characters (characters from a to Z, digits from 0-9, and the underscore _ character)
	# \s - Returns a match where the string contains a white space character
	df['preprocesstweet'] = df['preprocesstweet'].str.replace('[^\w\s]','', regex=True) 

	#df = df['preprocesstweet'] 
	#df = df.to_frame()
	df = df[df['preprocesstweet'].str.strip().astype(bool)]
	df.to_csv('preprocessed_' + search_term + '_tweets_data.csv', index = False)


#removing stopwords
#	nltk.download('stopwords') 
#	from nltk.corpus import stopwords
#	stop_words = stopwords.words('english')
#	newStopWords = ["day","hour","month","days","months","hours"] #adding more stopwords
#	stop_words.extend(newStopWords)
#	df['preprocesstweet'] = df['preprocesstweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
