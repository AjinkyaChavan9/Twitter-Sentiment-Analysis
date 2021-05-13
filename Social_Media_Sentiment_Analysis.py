import streamlit as st
import twint
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import preprocess
import VADER
import TextBlobModel
import LogisticRegression
import SVM
import NaiveBayes



def Visualize(model):
	df = pd.read_csv(model + '_Sentiments.csv')
	graph, ax = plt.subplots()
	ax.hist(df['Polarity'], align='right')
	st.pyplot(graph)
	if(model == 'TextBlob'):
		fig1, ax = plt.subplots(figsize=(8,6)) 
		for i in range(0, df.shape[0]):
			ax.scatter(df["Polarity_Score"][i], df["Subjectivity"][i], color='Blue')
		ax.set_title('Sentiment Analysis') 
		ax.set_xlabel('Polarity')
		ax.set_ylabel('Subjectivity')
		st.pyplot(fig1) 

	# word cloud visualization
	plt.style.use('fivethirtyeight')
	allWords = ' '.join([twts for twts in df['tweet']])
	wordCloud = WordCloud(width=1000, height=500, random_state=21, max_font_size=110).generate(allWords)
	plt.imshow(wordCloud, interpolation="bilinear")
	plt.axis('off')
	fig = plt.figure()
	st.image(wordCloud.to_array())

	st.dataframe(df)
	

Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
    </style> 
    
    <div class="title">
        <h1>Social Media Sentiment Analysis</h1>
    </div>
    """
st.markdown(Title_html, unsafe_allow_html=True) #Title rendering

## User Input
search_term = st.text_input("Enter Search Term to Analyze Tweets:")
no_of_tweets = st.text_input("Enter Number of Tweets:")

# Configure Twint
c = twint.Config()
c.Search = search_term
c.Lang = "en"
c.Limit = no_of_tweets
c.Store_csv = True
c.Custom["tweet"] = ["username", "tweet"]
c.Output = search_term + '_tweets_data.csv'

if st.button("Generate Dataset"):
	#Run Twint
	twint.run.Search(c)
	df = pd.read_csv(search_term + '_tweets_data.csv')
	st.success('Tweets Dataset Generated Successfully!')
	preprocess.preprocess_function(search_term)


selected_model = st.selectbox("Select Model for Sentiment Analysis: ", ['VADER (Valence Aware Dictionary for Sentiment Reasoning) Model', 'TextBlob', 'Logistic Regression', 'SVM', 'Naive Bayes'])

if st.button('Perform Sentiment Analysis'):
	if(selected_model == 'VADER (Valence Aware Dictionary for Sentiment Reasoning) Model'):
		VADER.predict_sentiment_VADER(search_term)
		Visualize('Vader')
	elif(selected_model == 'TextBlob'):
		TextBlobModel.predict_sentiment_TextBlob(search_term)
		Visualize('TextBlob')
	elif(selected_model == 'Logistic Regression'):
		LogisticRegression.predict_sentiment_LogReg(search_term)
		Visualize('Logistic_Regression')
	elif(selected_model == 'SVM'):
		SVM.predict_sentiment_SVM(search_term)
		Visualize('SVM')
	elif(selected_model == 'Naive Bayes'):
		NaiveBayes.predict_sentiment_NB(search_term)
		Visualize('NB')



	
