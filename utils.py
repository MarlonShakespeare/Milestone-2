import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import spacy
spacy_model = spacy.load('en_core_web_lg')

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

import gzip
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scipy.stats import skew, kurtosis
# nltk.download('stopwords')
stop_words = stopwords.words('english')


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
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n




#######################################################################################
############################### Feature Extraction ####################################
#######################################################################################

# topic extraction model
def extract_topic_feature(row, components=None, tokenizer=None, random_state=None):
    if components is None:
        components = len(row)
    if row is not None and components > 1:
        tweets = np.array(row)

        vectorize = TfidfVectorizer(tokenizer=tokenizer, 
                                    ngram_range=(1, 2),
                                    stop_words=None if tokenizer is not None else 'english',
                                    min_df=1)
        nmf_model = NMF(n_components=components, 
                        init='nndsvd', 
                        max_iter=100000, 
                        random_state=random_state)

        nmf_pipe = make_pipeline(vectorize, nmf_model)
        
        
        W = nmf_pipe.fit_transform(tweets)
        
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
    df.loc[:, 'topic_gini'] = df['topic_dist'].apply(lambda x: gini(x) if x is not None else np.nan)
    df.loc[:, 'std'] = df['topic_dist'].apply(lambda x: x.std() if x is not None else np.nan)
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
    return result_df

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
