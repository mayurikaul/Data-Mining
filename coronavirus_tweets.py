# Part 3: Text mining.

import pandas as pd
import nltk
import requests
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
    df = pd.read_csv(data_file, encoding='latin-1')
    return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    sentiments = df['Sentiment'].unique()
    return list(sentiments)

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    second = df['Sentiment'].value_counts(ascending = False).nlargest(2).index[1]
    return second

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    extremely_pos = df[df['Sentiment'] == 'Extremely Positive']
    counts = extremely_pos.groupby('TweetAt').size()
    max_date = counts.idxmax()
    return max_date


# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(str.lower)

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.replace('[^a-zA-Z ]', ' ')

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: ' '.join(x.split()))


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: nltk.word_tokenize(x))
    

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    word_count = tdf['OriginalTweet'].apply( lambda x: len(x))
    return word_count.sum()
	

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    outcome = set()
    tdf['OriginalTweet'].apply(outcome.update)
    return len(outcome)

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
    most_freq = pd.Series(' '.join(tdf['OriginalTweet'].apply(lambda x: ' '.join(x))).split()).value_counts()[:k]
    return most_freq.index.to_list()


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    
    url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
    stopwords = requests.get(url).content.decode('utf-8').split( "\n" )

    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [item for item in x if item not in stopwords])
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x : [item for item in x if len(item)>2])

    

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    stemmer = SnowballStemmer("english")
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x : [stemmer.stem(y) for y in x] )
	

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 

def mnb_predict(df):
    original_tweets = df['OriginalTweet']
    original_tweets = original_tweets.to_numpy()
    sentiments = df['Sentiment']
    sentiments = sentiments.to_numpy()
        
    vectorizer = CountVectorizer(max_df=0.7, min_df=0, max_features=20000)
    x = vectorizer.fit_transform(original_tweets)
    td = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

    model= MultinomialNB()
    model.fit(td, sentiments)
    sentiment_predict = model.predict(td)
    
    return sentiment_predict
	

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
    count = 0.0
    for i in range(len(y_true)):
        if (y_pred[i] == y_true[i]):
            count += 1

    accuracy = (count/len(y_true)) 
    return round(accuracy,3)










