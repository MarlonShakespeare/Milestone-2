import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import numpy as np
# nltk.download('stopwords')
stop_words = stopwords.words('english')




stemmer = PorterStemmer()
def stem_token(text):
    tokens = [stemmer.stem(word) for word in re.findall(r'\b\w\w+\b', text) if word not in stop_words]
    return tokens



def change_tweets_to_doc(row):
    if row is not None:
        return ' '.join(row)
    else:
        return np.nan


def prep_data_to_doc(df):
    _df = df.copy()
    _df.loc[:, 'tweet'] = _df['tweet'].apply(change_tweets_to_doc)
    return _df


def prep_data_explode(df):
    _df = df.copy()
    _df.loc[:, 'tweet'] = _df.explode('tweet')
    return _df


# using regex expression from https://www.geeksforgeeks.org/extract-urls-present-in-a-given-string/
def extract_num_links(tweet, unique=False):
    # remove comma or peroid as they throw off the regex
    links = re.findall(r'\b((?:https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:, .;]*[-a-zA-Z0-9+&@#/%=~_|])', tweet)
    if links:
        links = [link.split() for link in links][0]
        links = [link for link in links if (link.startswith('http') or link.startswith('ftp') or link.startswith('file'))]
    if unique:
        return len(set(links))
    else:
        return len(links)


def extract_num_mentions(tweet, unique=False):
    mentions = re.findall(r'@\w+', tweet)
    if unique:
        return len(set(mentions))
    else:
        return len(mentions)

def is_retweet(tweet):
    return 1 if tweet.startswith('RT') else 0


def tweet_level_prep(original_tweet_df, agg='mean'):
    tweet_df = prep_data_explode(original_tweet_df)
    tweet_df.loc[:, 'num_mentions'] = tweet_df['tweet'].apply(extract_num_mentions)
    tweet_df.loc[:, 'num_links'] = tweet_df['tweet'].apply(extract_num_links)
    tweet_df.loc[:, 'is_retweet'] = tweet_df['tweet'].apply(is_retweet)
    if agg == 'mean':
        return tweet_df[['ID', 'num_mentions', 'num_links', 'is_retweet']].groupby(['ID']).mean().reset_index(drop=True)
    elif agg == 'sum':
        return tweet_df[['ID', 'num_mentions', 'num_links', 'is_retweet']].groupby(['ID']).sum().reset_index(drop=True)
    else:
        return tweet_df