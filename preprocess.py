import pandas as pd
import os
import re
from collections import defaultdict
import numpy as np
from numpy.random import default_rng
import networkx as nx
import pickle
import json
import dgl
import scipy.sparse as sp
import torch
from tqdm import tqdm

tqdm.pandas(desc="my bar!")

#import fasttext.util
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

# Seed: 42
# USE THIS RNG ONLY (FOR REPRODUCIBILITY PURPOSES)
DEFAULT_RNG = default_rng(42)

### DATA LOADING ####

abs_path = '/home/indy-stg1/lobbymeptweets/'

def get_users(dir1):
    """returns  list of users in directory lobbytweetsep8 or meptweetsep8"""
    list_of_users = list()
    tweet_files = os.listdir(abs_path + dir1)
    for dir2 in tweet_files:
        list_of_users.append(dir2[:-13].lower())
    return list_of_users


def count_tweets_by_language():
    """returns a dictionary where the keys are the languages and the values are the number of tweets in the given language"""
    #we shuld exclude retweets
    lobby_tweets_by_language = defaultdict(int)
    mep_tweets_by_language = defaultdict(int)
    
    #count lobby tweets
    tweet_files = os.listdir(abs_path + 'lobbytweetsep8')
    for dir2 in tqdm(tweet_files):
        df = pd.read_json(abs_path + 'lobbytweetsep8'+'/'+dir2, lines=True)
        for data in df['data']:
            for i, d in enumerate(data):
                if d['text'][:2] != 'RT':
                    language = d['lang']
                    lobby_tweets_by_language[language] += 1
    
    #count mep tweets
    tweet_files = os.listdir(abs_path + 'meptweetsep8')
    for dir2 in tqdm(tweet_files):
        df = pd.read_json(abs_path + 'meptweetsep8'+'/'+dir2, lines=True)
        for data in df['data']:
            for i, d in enumerate(data):
                if d['text'][:2] != 'RT':
                    language = d['lang']
                    mep_tweets_by_language[language] += 1
    
    return lobby_tweets_by_language, mep_tweets_by_language


def get_text_by_id(language='en', mep_or_lobby='lobby'):
    """
    returns a list of dictionaries where a dictionary contains the tweet_id and the text of the tweet
    
    params:
    language: filters the tweet for the desired language
    mep_or_lobby: binary value 'mep' or 'lobby' will consider only tweets in lobby dir or tweets in mep dir
    """
    list_of_text_by_id = []
    
    if mep_or_lobby == 'lobby':
        dir1 = 'lobbytweetsep8'
        dummy = 1
    elif mep_or_lobby == 'mep':
        dir1 = 'meptweetsep8'
        dummy = 0
    else:
        raise Exception("Unknown directory for list_of_lobbies/meps")
    
   
    tweet_files = os.listdir(abs_path+dir1)
    for dir2 in tqdm(tweet_files):
        df = pd.read_json(abs_path + dir1+'/'+dir2, lines=True)
        for data in df['data']:
            for i, d in enumerate(data):
                if d['text'][:2] != 'RT':
                    if d['lang'] == language:
                        text_by_id = {}
                        text_by_id['id'] = d['id']
                        text_by_id['lobby'] = dummy
                        text_by_id['text'] = d['text']
                        text_by_id['user'] = dir2[:-13]
                        list_of_text_by_id.append(text_by_id) 
                    
    return list_of_text_by_id


##### NLP #####

def preprocess_data(text):
    """text preprocessing pipeline with casfolding, lemmatizer, tokanizer and puctuation and stop word removal"""
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer =  TweetTokenizer()
    stop_words = set(stopwords.words('english'))
 
    def lemmatize_text(text):
        return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((text))]
    
    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', (word))
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    def remove_stop_words(text, stop_words):
        return [item for item in text if item not in stop_words]
    
    
    text = lemmatize_text(text)
    text = remove_punctuation(text)
    text = remove_stop_words(text, stop_words)
    return text

def process_wrapper(lang):
    """wrapper function around processor steps"""
    #get all tweets
    text_mep = get_text_by_id(language=lang, mep_or_lobby ='mep')
    text_lobby = get_text_by_id(language=lang, mep_or_lobby ='lobby')
    df_mep = pd.DataFrame(text_mep)
    df_lobby = pd.DataFrame(text_lobby)
    
    #apply npl preprocessing pipeline
    df_mep['processed'] = df_mep['text'].progress_apply(lambda x: preprocess_data(x))
    df_lobby['processed'] = df_lobby['text'].progress_apply(lambda x: preprocess_data(x))
    
    df_mep = embed_dataframe_with_fasttext(df_mep)
    df_lobby = embed_dataframe_with_fasttext(df_lobby)
    
    return df_mep, df_lobby

##### EMBEDDING HELPER FUNCTIONS #####

def embed_dataframe_with_fasttext(df):
    """returns the 300 dimensional tweet embeddings (sum of corresponding word embeddings) using fasttext"""
    ft = fasttext.load_model('cc.en.300.bin')
    df['fasttext'] = df['text'].progress_apply(lambda x: ft.get_word_vector(x)) 
    return df

def get_weight(mep_embed, lobby_embed):
    """returns the maximum inner product value between two users"""
    return np.amax(mep_embed @ lobby_embed.T)
    
def get_embedding(node, df):
    """returns tweet embedding for a given user"""
    X = df[df['user'] == node]['fasttext'].to_numpy()
    X = np.stack(X, axis=0)[0]
    return X

def get_embedding_by_user(df_mep, df_lobby):
    """returns dataframes which contains all the tweet embeddings per user"""
    df_mep_users = pd.DataFrame()
    df_lobby_users = pd.DataFrame()
    df_mep_users['user'] = df_mep['user'].unique()
    df_lobby_users['user'] = df_lobby['user'].unique()
    df_mep_users['fasttext'] = df_mep_users['user'].progress_apply(lambda x: get_embedding(x, df_mep))
    df_lobby_users['fasttext'] = df_lobby_users['user'].progress_apply(lambda x: get_embedding(x, df_lobby))
    return df_mep_users, df_lobby_users

def average_embedding_by_user(df_mep_users, df_lobby_users):
    """returns dataframes which contain the average embeddings for a given user"""
    df_mep_users['fasttext'] = df_mep_users['fasttext'].progress_apply(lambda x: x.sum(axis=0))
    df_lobby_users['fasttext'] = df_lobby_users['fasttext'].progress_apply(lambda x: x.sum(axis=0))
    return df_mep_users, df_lobby_users
    
    
##### PICKLE & JSNON #####
    
def dump_pickle(data, file):
    with open(file, 'wb') as handle:
        pickle.dump(data , handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_pickle(file):
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def dump_json(data, file):
    with open(file, 'w') as handle:
        json.dump(data, handle)
    
def load_json(file):
    with open(file) as handle:
        data = json.load(handle)
    return data


def train_test_split(D):
    """
    params:
    D: dgl graph object containing all retweet links as edges and node embeddings
    
    return:
    train_g: dgl graph with all train edges
    train_pos_g: torch tensor - indexes of train edges
    train_neg_g: torch tensor - indexes of train non-edges
    test_pos_g: torch tensor - indexes of test edges
    test_neg_g: torch tensor - indexes of test non-edges

    """
    
    # Split edge set for training and testing
    u, v = D.edges()
    eids = np.arange(D.number_of_edges())
    eids = DEFAULT_RNG.permutation(eids)
    edge_dict = {(int(u[i]), int(v[i])): i for i in eids}
    test_size = int(len(eids) * 0.2)
    train_size = D.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(D.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = DEFAULT_RNG.choice(len(neg_u), D.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_neg_u, train_neg_v = torch.tensor(train_neg_u), torch.tensor(train_neg_v)

    test_indexes = []
    for i in range(len(test_pos_u)):
        test_indexes.append(edge_dict[(int(test_pos_v[i]), int(test_pos_u[i]))])
        test_indexes.append(edge_dict[(int(test_pos_u[i]), int(test_pos_v[i]))])
    
    train_g = dgl.remove_edges(D, np.array(test_indexes))

    train_pos = torch.cat((train_pos_u, train_pos_v)), torch.cat((train_pos_v, train_pos_u))
    train_neg = torch.cat((train_neg_u, train_neg_v)), torch.cat((train_neg_v, train_neg_u))

    train_pos_g = dgl.graph(train_pos, num_nodes=D.number_of_nodes())
    train_neg_g = dgl.graph(train_neg, num_nodes=D.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=D.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=D.number_of_nodes())
    
    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g


def plot_roc(classifiers, N):
    """classifiers is a dictionary key=name, value=numpy array of predictions"""
    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
    for clss, y in classifiers.items():
    
        y_test = np.concatenate((np.ones(N), np.zeros(N)))
        fpr, tpr, _ = roc_curve(y_test,  y)
        auc = roc_auc_score(y_test, y)
    
        result_table = result_table.append({'classifiers':clss,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    fig , ax = plt.subplots(ncols=1, nrows=1, figsize=(8,6))

    for i in result_table.index:
        ax.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
   
    ax.set(ylabel="True Positive Rate", xlabel="False Positive Rate", title="Receiver operating characteristic")
    ax.legend(prop={'size':13}, loc='lower right')

    plt.show()
