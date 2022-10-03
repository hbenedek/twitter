import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
import community as community_louvain
from netgraph import Graph

import gensim
import nltk
import string
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('wordnet')      
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

nltk.download('stopwords')

#Colers used to plot clusters
community_to_color = {
    0 : 'C0',
    1 : 'C1',
    2 : 'C2',
    3 : 'C3',
    4 : 'C4',
    5 : 'C5',
    6 : 'C6',
    7 : 'C7',
    8 : 'C8',
    9 : 'C9',
    10 : 'C10',
    11 : 'C11',
    12 : 'C12',
    13 : 'C14',
    14 : 'C15',
    15 : 'C16',
    16 : 'C13',
    17 : 'C17'
}


"""
fig, ax = plt.subplots(figsize=(20, 20))
node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

Graph(G,
      node_color=node_color, node_edge_width=0.1, edge_alpha=0.1,
      node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community)
       ,edge_layout_kwargs=dict(k=2000),
)

plt.show()
"""


def get_cluster_nodes(cluster_num):
    """ 
    @param
    cluster_num: int: the cluster number
    @return
    A list of nodes in the cluster cluster_num
    """
    ls = [ key for (key,value) in node_to_community.items() if value  == cluster_num]
    return ls


def get_cluster_edges(cluster_num):
    """ 
    @param
    cluster_num:int: the cluster number
    @return
    A list of edges in the cluster cluster_num
    """
    nodes=get_cluster_nodes(cluster_num)
    edges=[edge for edge in G.edges if ((edge[0] in nodes) and (edge[1] in nodes))]
    return edges


def get_cluster_tweets(cluster_num,df):
    """ 
    @param
    cluster_num : int: the cluster number
    df: tweets dataframe
    @return
    A list of tweets(String) in the cluster cluster_num
    """
    ids=[]
    edges=get_cluster_edges(cluster_num)
    for edge in edges:
        ids.append(G.edges[edge]['tweetRef']['id'])
    
    return df.loc[df.index.intersection(ids)].text.tolist()




stopwords = set(stopwords.words('english'))

exclude = set(string.punctuation)

lemma = WordNetLemmatizer()


def clean(document):
    """ 
    @param
    document: string
    @return : string
    Clean document (remoove stopwords and punctuation and the lemmatize)
    """

    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stopwords])

    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)

    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())

    return normalized


def topic_per_cluster(num_cluster):
    """ 
    @param
    num_cluster: int the cluster number 
    @return : List(string)
     extract the topic from the cluster num_cluster
    """
    cluster_tweets=get_cluster_tweets(num_cluster)
    clean_tweets = [clean(tweet).split() for tweet in cluster_tweets]
    dictionary = corpora.Dictionary(clean_tweets)
    DT_matrix = [dictionary.doc2bow(tweet) for tweet in clean_tweets]
    Lda_object = gensim.models.ldamodel.LdaModel
    lda_model_1 = Lda_object(DT_matrix, num_topics=10, id2word = dictionary)
    return lda_model_1.print_topics(num_topics=10, num_words=5)
