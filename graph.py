import networkx as nx
import pandas as pd
import numpy as np
import os

def build_graph(main_dir = '../lobbymeptweets/', lobby_dir = 'lobbytweetsep8', mep_dir = 'meptweetsep8'):
    G = nx.MultiGraph()

    # Collect all Twitter handles
    # assumed format: <handle>tweetsep8.txt
    tweet_dirs = os.listdir(main_dir)
    list_of_lobbies = list()
    list_of_meps = list()
    for dir1 in tweet_dirs:
        tweet_files = os.listdir(main_dir+dir1)
        for dir2 in tweet_files:
            if dir1 == lobby_dir:
                list_of_lobbies.append(dir2[:-13].lower())
            elif dir1 == mep_dir:
                list_of_meps.append(dir2[:-13].lower())
            else:
                raise Exception("Unknown directory for list_of_lobbies/meps")

    # Useful stats for execution analysis
    count_global_tweets = 0
    count_global_missing_references = 0
    count_global_missing_tweets = 0
    count_global_missing_users = 0
    count_global_skipped_tweets_users = 0

    for dir1 in tweet_dirs:
        tweet_files = os.listdir(main_dir+dir1)
        for dir2 in tweet_files:
            df = pd.read_json(main_dir+dir1+'/'+dir2, lines=True)
            for _, row in df.iterrows():
                tweets = row.data
                try:
                    tweets_includes = pd.DataFrame(row.includes['tweets'])
                except:
                    count_global_missing_references += 1
                    continue
                    
                tweets_includes.set_index('id', inplace=True)
                tweets_includes = tweets_includes[~tweets_includes.index.duplicated(keep='first')]
                users_includes = pd.DataFrame(row.includes['users'])
                users_includes.set_index('id', inplace=True)
                users_includes = users_includes[~users_includes.index.duplicated(keep='first')]
                
                for tweet in tweets:
                    if 'referenced_tweets' not in tweet.keys():
                        count_global_missing_references += 1
                        continue
                        
                    own_id = tweet['author_id']
                    own_name = users_includes.username[own_id]
                    tweet_id = tweet['id']
                    
                    for reference in tweet['referenced_tweets']:
                        count_global_tweets += 1
                        referenced_id = reference['id']
                        try:
                            referenced_tweet = tweets_includes.loc[referenced_id]
                        except:
                            # From quick check - removed tweet
                            count_global_missing_tweets += 1
                            continue
                            
                        referenced_user_id = referenced_tweet['author_id']
                        referenced_user_name = users_includes.username[referenced_user_id]
                            
                        if (dir1 == lobby_dir and referenced_user_name.lower() not in list_of_meps):
                           count_global_skipped_tweets_users += 1
                           continue
                        if (dir1 == mep_dir and referenced_user_name.lower() not in list_of_lobbies):
                           count_global_skipped_tweets_users += 1
                           continue
                        
                        # Data to be put on nodes/edges
                        own_isMep = own_name.lower() in list_of_meps
                        own_isLobby = own_name.lower() in list_of_lobbies
                        referenced_isMep = referenced_user_name.lower() in list_of_meps
                        referenced_isLobby = referenced_user_name.lower() in list_of_lobbies
                        referenced_tweet_language = referenced_tweet['lang']
                        is_retweet = reference['type'] == 'retweeted'

                        try:
                            G.add_node(own_name, isMep = own_isMep, isLobby = own_isLobby)
                            G.add_node(referenced_user_name, isMep = referenced_isMep, isLobby = referenced_isLobby)
                            G.add_edge(own_name, referenced_user_name, originalId = tweet_id, tweetRef=reference, source = dir1 + "/" + dir2, \
                                    lang = referenced_tweet_language, isRetweet = is_retweet)
                        except Exception as e:
                            print(f"Failed to create edge with values ({own_name}, {referenced_user_name})")
                            raise e
                            
    print(f"""
    Total number of tweet references considered: {count_global_tweets}

    Total errors:
        {count_global_missing_references} missing tweet reference data
        {count_global_missing_tweets} missing tweets in existing references
        {count_global_missing_users} missing users in existing references
        {count_global_skipped_tweets_users} skipped tweet references due to irrelevant users
        """)

    nx.write_gpickle(G, "graph.pickle")
    return G

def build_bipartite_edges(df_mep_users, df_lobby_users):
    """retruns networkx graph with edge weights calculated from inner product embeddings"""
    B = nx.Graph()
    mep = list(df_mep_users['user'])
    lobby = list(df_lobby_users['user'])
    B.add_nodes_from(mep, bipartite=0)
    B.add_nodes_from(lobby, bipartite=1)
    mep_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    lobby_nodes = set(B) - mep_nodes
    
    for mep_node in tqdm(mep_nodes):
        for lobby_node in lobby_nodes:
            mep_ft = df_mep_users[df_mep_users['user'] == mep_node]['fasttext'].values[0]
            lobby_ft = df_lobby_users[df_lobby_users['user'] == lobby_node]['fasttext'].values[0]
            weight = get_weight(mep_ft, lobby_ft)
            B.add_weighted_edges_from([(mep_node, lobby_node, weight)])
    return B

def graph_to_edge_dataframe(B, G):
    """
    params:
    G: bipartite graph containing retweet edges
    B: bipartite graph containing all extracted tweet weights from embeddings (with some aggregation method)
    
    return:
    data: dataframe containing all edge features and labels
    """
    # making sure all mep and lobby users are casefolded in both graphs
    mapping = {n: n.lower() for n in B.nodes}
    B = nx.relabel_nodes(B, mapping)
    
    # drop meps and lobbies where the two sets not overlap
    G_mep = set([n.lower() for n,d in G.nodes(data=True) if d['isMep'] == 1])
    B_mep = set([n.lower() for n,d in B.nodes(data=True) if d['bipartite'] == 0])
    G_lobby = set([n.lower() for n,d in G.nodes(data=True) if d['isMep'] == 0])
    B_lobby = set([n.lower() for n,d in B.nodes(data=True) if d['bipartite'] == 1])
    valid_meps = G_mep & B_mep
    valid_lobby = G_lobby & B_lobby
    
    df_list = []
    for (mep, lobby) in tqdm(list(B.edges)):
        if (mep.lower() in valid_meps) & (lobby.lower() in valid_lobby):
            weight = B[mep][lobby]["weight"]
            if G.has_edge(mep, lobby):
                df_list.append({'mep': mep, 'lobby': lobby, 'weight': weight ,'retweet': 1})
            else:
                df_list.append({'mep': mep, 'lobby': lobby, 'weight': weight ,'retweet': 0})
    df = pd.DataFrame(df_list)
    return df




def build_dgl_graph(df_mep_users, df_lobby_users, G):
    """
    params:
    df_mep_users: dataframe containing user embeddings for MEPs
    df_lobby_users: dataframe containing user embeddings for lobbies
    G: networkx graph containing all retweet links as edges

    return:
    D: dgl graph object, with 300 dimensional tweet embedding node feature
    """
    #average embedding vectors to get node features
    df_mep_users['fasttext'] = df_mep_users['fasttext'].progress_apply(lambda x: x.sum(axis=0))
    df_lobby_users['fasttext'] = df_lobby_users['fasttext'].progress_apply(lambda x: x.sum(axis=0))
    
    #make all username lowercase
    df_mep_users['user'] = df_mep_users['user'].progress_apply(lambda x: x.lower())
    df_lobby_users['user'] = df_lobby_users['user'].progress_apply(lambda x: x.lower())
    mapping = {node: node.lower() for node in G.nodes}
    G = nx.relabel_nodes(G, mapping)
    
    #get intersection of G graph nodes and df 'users'
    G_mep = set([n for n,d in G.nodes(data=True) if d['isMep'] == 1])
    mep = set(df_mep_users['user'].unique())
    G_lobby = set([n for n,d in G.nodes(data=True) if d['isLobby'] == 1])
    lobby = set(df_lobby_users['user'].unique())
    valid_meps = G_mep & mep
    valid_lobby = G_lobby & lobby
    print(f'valid meps: {len(valid_meps)}, valid lobby: {len(valid_lobby)}')
    
    #build networkx graph object from multigraph object
    G1 = nx.Graph()
    for u,v,data in G.edges(data=True):
        if G1.has_edge(u,v):
            pass
        else:
            G1.add_edge(u, v)
    
    #merge all edges and node features on a new graph
    nx_g = nx.Graph()
    nodes_by_index = {}
    for i, lobby_node in enumerate(list(valid_lobby)):
        lobby_ft = df_lobby_users[df_lobby_users['user'] == lobby_node]['fasttext'].values[0]
        nx_g.add_nodes_from(np.array([i]), name=lobby_node, feat1=lobby_ft, isMep=0)
      
    for j, mep_node in enumerate(list(valid_meps)):
        mep_ft = df_mep_users[df_mep_users['user'] == mep_node]['fasttext'].values[0]
        nx_g.add_nodes_from(np.array([j+i]), name=mep_node, feat1=mep_ft, isMep=1)
        
    
    for lobby_node in list(valid_lobby):
        for mep_node in list(valid_meps):
            if G1.has_edge(mep_node, lobby_node):
                nx_g.add_edge(nodes_by_index[mep_node], nodes_by_index[lobby_node])
                
    #convert to DGL object
    D = dgl.from_networkx(nx_g, node_attrs=['feat1'])
    
    return D