import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
spacy_model = spacy.load('en_core_web_lg')
nlp = spacy.load('en_core_web_md')

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import gzip
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scipy.stats import skew, kurtosis
# nltk.download('stopwords')
stop_words = stopwords.words('english')

import random as rnd
import tensorflow as tf
from tensorflow import keras

###################################################################################
# stemming and lemmatizing
###################################################################################

def stem_token(text):
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in re.findall(r'\b\w\w+\b', text) if word not in stop_words]
    return tokens

def spacy_tokenizer(text):
    sp_text = spacy_model(text)
    tokens = [token.lemma_ for token in sp_text if (not token.is_stop and token.is_alpha)]
    return tokens


####################################################################################
################################ helper functions ##################################
####################################################################################

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
    _df = _df.explode('tweet')
    return _df

# metrics
# unweighted gini coefficient measure taken from 
# https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python 
# by user Gaëtan de Menten
def gini(x):
    if np.array(x).shape == (0,):
        return np.nan
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n




#######################################################################################
############################### Feature Extraction ####################################
#######################################################################################

# topic extraction model
def extract_topic_feature(row, components=None, tokenizer=None, random_state=None):
    
    if row is not None:
        tweets = np.array(row)

        vectorize = TfidfVectorizer(tokenizer=tokenizer, 
                                    ngram_range=(1, 2),
                                    stop_words=None if tokenizer is not None else 'english',
                                    min_df=2,
                                    max_features=3000)
        try:

            vect_tweets = vectorize.fit_transform(tweets)
        except Exception:
            return np.nan

        if components is None:
            components = min(vect_tweets.shape)
            if components < 2:
                return np.nan
        elif components > min(vect_tweets.shape):
            components = min(vect_tweets.shape)
            if components < 2:
                return np.nan

        nmf_model = NMF(n_components=components, 
                        init='nndsvd', 
                        max_iter=200, 
                        random_state=random_state)
        
        try:
            W = nmf_model.fit_transform(vect_tweets)
        except ValueError as e:
            print(e)
            return np.nan
        
        index_max = []
        for index in range(W.shape[0]):
            max_val_index = np.argmax(W[index])
            index_max.append(max_val_index)
            
        index_norm = np.array(index_max) / components
        
        return index_norm
    else:
        return np.nan


# import this one
# topic extraction and metrics feature generation
def extract_nmf_feature(df, tokenizer=spacy_tokenizer):
    df.loc[:, 'topic_dist'] = df['tweet'].apply(lambda x: extract_topic_feature(x, tokenizer=tokenizer) if x is not None else np.nan)
    df.loc[:, 'topic_skew'] = df['topic_dist'].apply(lambda x: skew(x) if x is not None else np.nan)
    df.loc[:, 'topic_kurtosis'] = df['topic_dist'].apply(lambda x: kurtosis(x) if x is not None else np.nan)
    df.loc[:, 'topic_gini'] = df['topic_dist'].apply(lambda x: gini(x) if (x is not None) and (len(x) > 0) else np.nan)
    df.loc[:, 'topic_std'] = df['topic_dist'].apply(lambda x: np.array(x).std() if x is not None else np.nan)
    return df



# number of links
# using regex expression from 
# https://www.geeksforgeeks.org/extract-urls-present-in-a-given-string/
def extract_num_links(tweet, unique):
    # remove comma or peroid as they throw off the regex
    links = re.findall(r'\b((?:https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:, .;]*[-a-zA-Z0-9+&@#/%=~_|])', tweet)
    if links:
        links = [link.split() for link in links][0]
        links = [link for link in links if (link.startswith('http') or link.startswith('ftp') or link.startswith('file'))]
    if unique:
        return len(set(links))
    else:
        return len(links)

# number of mentions
def extract_num_mentions(tweet, unique):
    mentions = re.findall(r'@\w+', tweet)
    if unique:
        return len(set(mentions))
    else:
        return len(mentions)

# is retweet
def is_retweet(tweet):
    return 1 if tweet.startswith('RT') else 0

# import this one
# tweet level feature generation links, mentions, retweet
def tweet_level_feature_generation(original_tweet_df, normalize=False, unique=False):
    tweet_df = prep_data_explode(original_tweet_df)

    mentions_col = 'num_mentions' if not unique else 'num_mentions_unique'
    links_col = 'num_links' if not unique else 'num_links_unique'

    tweet_df.loc[:, mentions_col] = tweet_df['tweet'].apply(extract_num_mentions, unique=unique)
    tweet_df.loc[:, links_col] = tweet_df['tweet'].apply(extract_num_links, unique=unique)
    tweet_df.loc[:, 'retweet'] = tweet_df['tweet'].apply(is_retweet)
    tweet_df['num_tweets'] = 1

    result_df = tweet_df[['ID', mentions_col, links_col, 'retweet', 'num_tweets']].groupby(['ID']).sum().reset_index()
    if normalize:
        for col in result_df.columns[1:-1]:
            result_df[col] = result_df[col] / result_df.iloc[:, -1]
    return original_tweet_df.merge(result_df, how='inner', on='ID')

################################################################################################
##################################### data splitting ###########################################
################################################################################################

def train_test_dev_split(df, train, dev_test_split=None, shuffle=True, random_state=None):
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train, shuffle=shuffle, random_state=random_state)
    train_df = pd.DataFrame(pd.concat([X_train, y_train], axis=1), columns=df.columns)
    test_df = pd.DataFrame(pd.concat([X_test, y_test], axis=1), columns=df.columns)

    if dev_test_split is not None:
        X, y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, train_size=dev_test_split, shuffle=shuffle, random_state=random_state)
        dev_df = pd.DataFrame(pd.concat([X_dev, y_dev], axis=1), columns=df.columns)
        test_df = pd.DataFrame(pd.concat([X_test, y_test], axis=1), columns=df.columns)
        return train_df, dev_df, test_df

    else:
        return train_df, test_df

# Import this one
def split_all_data(train_split:float, train_path:str='Twibot-20/train.json', test_path:str='Twibot-20/test.json', 
                   dev_path:str='Twibot-20/dev.json', support_path:str='Twibot-20/support.json', 
                   include_support:bool=False, dev_test_split:float=None, shuffle:bool=True, 
                   save_result:bool=False, random_state:int=None):
    """loads the datafrom json files. Splits the data int train, test and optionally dev sets
    removes all dev and test entries from support to prevent leakage, and optionally saves the result

    Args:
        train_split (float): train ratio
        train_path (str, optional): path to train.json. Defaults to Twobot-20/train.json
        test_path (str, optional): path to test.json. Defaults to Twobot-20/test.json
        dev_path (str, optional): path to dev.json. Defaults to Twobot-20/dev.json
        support_path (str, optional): path to support.json. Defaults to Twobot-20/support.json
        include_support (bool, optional): weather to include support file or not. Default False
        dev_test_split (float, optional): ratio of dev set to test set. If None no dev set will be produced. Defaults to None.
        shuffle (bool, optional): whether to shuffle before splitting into sets recomended. Defaults to True.
        save_result (bool, optional): wether to save the resulting dataframes as csv's or retrun them. Defaults to False.
        random_state (int, optional): random seed to pass to sklearn.model_selection.train_test_split. Defaults to None

    Returns:
        dict of DataFrame: returns resulting dict of dataframes if include_support is False support will be None same with dev_test_split
    """
    results = {
        'train': None,
        'dev': None,
        'test': None,
        'support': None
    }

    # load in the data
    print('Loading the data...')
    train = pd.read_json(train_path)
    test = pd.read_json(test_path)
    dev = pd.read_json(dev_path)
    if include_support:
        results['support'] = pd.read_json(support_path)


    # combine train, test, dev
    twibot_df = pd.concat([train, dev, test], axis=0).reset_index(drop=True)

    # split the sets
    print('Splitting the data...')
    if dev_test_split is not None:
        results['train'], results['dev'], results['test'] = train_test_dev_split(twibot_df, 
                                                                                 train=train_split, 
                                                                                 dev_test_split=dev_test_split, 
                                                                                 shuffle=shuffle, 
                                                                                 random_state=random_state)
    else:
        results['train'], results['test'] = train_test_dev_split(twibot_df, 
                                                                 train=train_split, 
                                                                 dev_test_split=dev_test_split, 
                                                                 shuffle=shuffle, 
                                                                 random_state=random_state)

    # save results
    if save_result:
        print('Saving...')
        results['train'].to_csv('Twibot-20/train_labeled.csv', index=False)
        results['test'].to_csv('Twibot-20/test_labeled.csv', index=False)
        if dev_test_split is not None:
            results['dev'].to_csv('Twibot-20/dev_labeled.csv', index=False)
        if include_support:
            results['support'].to_csv('Twibot-20/support_with_no_test_or_dev.csv.gz', index=False)
    else:
        pass
    # return results
    return results


################################################################################################
##################################### deep learning ############################################
################################################################################################


def nn_tweet_only(train_df, val_df, layers=(100,100), epochs=20):
    """
    Arguments: dataframes of the type output by utils.split_all_data
        Layers sets the size of the two densely connected NN layers
        
    Returns: A fitted TFIDF vecorizer and a fitted NN model
    """

    train = train_df.copy()
    val = val_df.copy()
    
    # Change tweet list into single character string
    print("Prepping tweet data...")
    train['tweet'] = train['tweet'].apply(lambda x: ' '.join(x) if x is not None else np.nan)
    val['tweet'] = val['tweet'].apply(lambda x: ' '.join(x) if x is not None else np.nan)
    
    train.dropna(inplace=True)
    val.dropna(inplace=True)

    # Vectorize tweet data
    # Prepare vectors for the model
    print("Vectorizing tweets...")
    vectorizer = TfidfVectorizer(stop_words = 'english', min_df=500, ngram_range=(1,3))

    X_train = vectorizer.fit_transform(train.tweet).toarray()
    feat_size = X_train.shape[1]   
    
    y_train = np.array(train.label)
    
    X_val = vectorizer.transform(val.tweet).toarray()
    y_val = np.array(val.label)

    # Initialize the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(layers[0], activation='relu'),
        tf.keras.layers.Dense(layers[1], activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # Set up a callback to stop when loss on the validation set stops decreasing
    # Parameters were determined in the exploration phase
    val_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.005,
        patience=3,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                 )

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[val_stop])
    
    return vectorizer, model


def add_nn_signal(test_df, vectorizer, model):
    """
    Arguments: A dataframe of the type output by utils.split_all_data
               A TFIDF vectorizer and NN model of the types output by utils.nn_tweet_only 
               
    Returns: A dataframe with a new column 'nn_signal' which gives the model's prediction (as a probabilty)
                that the account is human (label==1 on original)  
    """
    
    test = test_df.copy()
    out_df = test_df.copy()
    
    test['tweet'] = test['tweet'].apply(lambda x: ' '.join(x) if x is not None else ' ')

    
    out_df['nn_signal'] = model.predict(vectorizer.transform(test.tweet).toarray()).T[1]
    
    return out_df
    

    

################################################################################################
##################################### adding features ##########################################
################################################################################################


def add_feat(df):
    """
    Adds tweet and account features to dataframe df
    
    Argument: dataframe of type output by utils.split_all_data
    Returns: dataframe of same type
    
    Assumes that file lang.csv is in home directory
    """
    
    # Number of languages represented in tweets
    lang_df = pd.read_csv('lang.csv')
    tw_features_df = df.merge(lang_df, how='inner', on='ID')

    # Min, max, average, and standard deviation of lengths of tweets
    tw_features_df['tweet_min_len'] = tw_features_df['tweet'].apply(lambda x: min([ len(t) for t in x ]))
    tw_features_df['tweet_max_len'] = tw_features_df['tweet'].apply(lambda x: max([ len(t) for t in x ]))
    tw_features_df['tweet_av_len'] = tw_features_df['tweet'].apply(lambda x: np.mean([ len(t) for t in x ]))
    tw_features_df['tweet_len_std'] = tw_features_df['tweet'].apply(lambda x: np.std([ len(t) for t in x ]))

    # Lengths of user and screen names
    tw_features_df['user_name_len'] = tw_features_df['profile'].apply(lambda x: len(x['name']))
    tw_features_df['screen_name_len'] = tw_features_df['profile'].apply(lambda x: len(x['screen_name']))

    # Number of distinct characters in user name
    tw_features_df['user_name_chars'] = tw_features_df['profile'].apply(lambda x: len(set(x['name'])))

    # Protected and verified status
    # ****** Only two protected accounts!
    tw_features_df['protected'] = tw_features_df['profile'].apply(lambda x: int(x['protected'] == 'True '))
    tw_features_df['verified'] = tw_features_df['profile'].apply(lambda x: int(x['verified'] == 'True '))

    # Is a URL associated with the account
    tw_features_df['has_url'] = tw_features_df['profile'].apply(lambda x: int(x['url'] != 'None '))

    # Social counts
    tw_features_df['followers_count'] = tw_features_df['profile'].apply(lambda x: int(x['followers_count']))
    tw_features_df['friends_count'] = tw_features_df['profile'].apply(lambda x: int(x['friends_count']))
    tw_features_df['favourites_count'] = tw_features_df['profile'].apply(lambda x: int(x['favourites_count']))

    ref_date = pd.to_datetime('May 01 2022')

    # How many days before May 1 2022 was the account created
    tw_features_df['days_old'] = tw_features_df['profile'].apply(lambda x: (ref_date - pd.to_datetime(x['created_at']).replace(tzinfo=None)).days)

    return tw_features_df


def calculate_similarity(df):
    "Takes a dataframe of tweets, calculates their similarity and appends them to the dataframe."
    similarity = []
    for user_sent in df:
        w2v = []
        # Ignoring users where tweet = None
        if user_sent is not None:
            # We take each user sentence, and get it's average word2vec embedding and append it to w2v list
            # For each user we will have a w2v array created, after all the users sentences are done, we calculate the 
            # cosine similarity between the w2v vectors and append the value to the similarity list, which will be later on
            # appended to the dataset as a feature in the supervised learning model
            for sentence in user_sent:
                vec = nlp(sentence).vector
                w2v.append(vec)
    
            sim = cosine_similarity(w2v)
            np.fill_diagonal(sim,0)
            similarity.append(sim.mean())
    
    
    return similarity

def tweet_level_features(df):
    df['listed_count'] = df['profile'].apply(lambda x: int(x['listed_count']))
    df['statuses_count'] = df['profile'].apply(lambda x: int(x['statuses_count']))
    df['description_len'] =  df['profile'].apply(lambda x: len(x['description']))
    return df

def generate_sentiment_feature(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['tweet'].apply(lambda x: np.mean([ sia.polarity_scores(t)['compound'] for t in x ]))
    df['sentiment'] = df['sentiment'].apply(lambda x: x>0) 
    df['sentiment'] = df['sentiment'].replace({True:'Positive',False:'Negative'})
    return df
    
